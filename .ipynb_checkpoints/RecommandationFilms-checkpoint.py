#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
import os
from collections import defaultdict
from kagglehub import dataset_download
pd.set_option("display.precision", 1)


# In[2]:


# Téléchargement du dataset (remplacez par votre méthode de téléchargement)

# Charger les fichiers séparément
ratings_for_additional_users = pd.read_csv("../data_release/ratings_for_additional_users.csv")
movies = pd.read_csv( "../data_release/movies.csv")
movie_elicitation_set = pd.read_csv( "../data_release/movie_elicitation_set.csv")
belief_data = pd.read_csv( "../data_release/belief_data.csv")
user_rating_history = pd.read_csv( "../data_release/user_rating_history.csv")
user_recommendation_history = pd.read_csv( "../data_release/user_recommendation_history.csv")

print(f"Données chargées:")
print(f"- Movies: {movies.shape}")
print(f"- ratings for additional users: {ratings_for_additional_users.shape}")
print(f"- user rating history: {user_rating_history.shape}")



# In[3]:


ratings_for_additional_users.head()


# In[4]:


movies.head()


# In[5]:


user_rating_history.head()


# In[6]:


# Extraire l'année avec regex
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

print("\nFilm le plus ancien :")
print(movies.loc[movies["year"].idxmin()])

print("\nFilm le plus récent :")
print(movies.loc[movies["year"].idxmax()])


# ## 3. Préparation et nettoyage des données

# In[7]:


min_ratings=20
ratings = user_rating_history


# In[8]:


# Filtrer les utilisateurs et films avec suffisamment de ratings
movie_counts = ratings['movieId'].value_counts()

valid_movies = movie_counts[movie_counts >= min_ratings].index

filtered_ratings = ratings[
    (ratings['userId']) & 
    (ratings['movieId'].isin(valid_movies))
]

print(f"Données filtrées: {len(filtered_ratings)} ratings")
print(f"Utilisateurs: {filtered_ratings['userId'].nunique()}")
print(f"Films: {filtered_ratings['movieId'].nunique()}")


# In[9]:


# Extraire la liste des genres
genres_list = []
for genres_str in movies['genres']:
    if pd.isna(genres_str) or genres_str == "(no genres listed)":
        continue
    genres_list.extend(genres_str.split('|'))

unique_genres = sorted(list(set(genres_list)))
print(f"Genres disponibles: {unique_genres}")


# In[10]:


# Créer des features de genres pour les films
movie_genre_features = pd.DataFrame(index=movies['movieId'])
for genre in unique_genres:
    movie_genre_features[f'genre_{genre}'] = 0

for idx, row in movies.iterrows():
    movie_id = row['movieId']
    if pd.isna(row['genres']) or row['genres'] == "(no genres listed)":
        continue
    genres = row['genres'].split('|')
    for genre in genres:
        if f'genre_{genre}' in movie_genre_features.columns:
            movie_genre_features.loc[movie_id, f'genre_{genre}'] = 1

# === Calcul des statistiques des films ===
movie_stats = filtered_ratings.groupby('movieId').agg({
    'rating': ['mean', 'count']
}).round(2)
movie_stats.columns = ['movie_rating_mean', 'movie_rating_count']

# Ajouter ces stats aux features de films
movie_genre_features = movie_genre_features.join(movie_stats, how='left')
movie_genre_features = movie_genre_features.fillna({'movie_rating_mean': 0, 'movie_rating_count': 0})


# In[11]:


movie_genre_features.head()


# In[12]:


# Créer un dictionnaire des films enrichi avec stats
movie_dict = {}
for _, row in movies.iterrows():
    movie_id = row['movieId']
    mean_rating = movie_stats.loc[movie_id, 'movie_rating_mean'] if movie_id in movie_stats.index else 0
    count_rating = movie_stats.loc[movie_id, 'movie_rating_count'] if movie_id in movie_stats.index else 0
    
    movie_dict[movie_id] = {
        'title': row['title'],
        'genres': row['genres'],
        'rating_mean': mean_rating,
        'rating_count': count_rating
    }


# In[13]:


movie_dict[1]


# In[14]:


# Calculer les statistiques utilisateur
user_stats = filtered_ratings.groupby('userId').agg({
    'rating': ['mean', 'count']
}).round(1)
user_stats.columns = ['rating_mean', 'rating_count']


# In[15]:


user_stats.head()


# In[16]:


# S'assurer que c'est bien en datetime
filtered_ratings = filtered_ratings.copy()
filtered_ratings["tstamp"] = pd.to_datetime(filtered_ratings["tstamp"])


# In[17]:


# ==============================
# === Stats pour les UTILISATEURS ===
# ==============================
user_time_diffs = []

for user_id, group in filtered_ratings.groupby("userId"):
    group_sorted = group.sort_values("tstamp")
    if len(group_sorted) > 1:
        deltas = group_sorted["tstamp"].diff().dt.total_seconds().dropna()
        avg_delta = deltas.mean()
    else:
        avg_delta = np.nan
    
    user_time_diffs.append((user_id, avg_delta))

user_time_diffs = pd.DataFrame(user_time_diffs, columns=["userId", "user_avg_time_diff"])

# Ajouter aux stats utilisateurs
user_stats = user_stats.join(user_time_diffs.set_index("userId"), how="left")
user_stats["user_avg_time_diff"] = user_stats["user_avg_time_diff"].fillna(-1)



# In[18]:


user_stats.head()


# In[19]:


# Créer des features de préférence de genre par utilisateur
user_genre_prefs = pd.DataFrame(index=user_stats.index)
for genre in unique_genres:
    user_genre_prefs[f'pref_{genre}'] = 0.0

# Calculer les préférences de genre basées sur les ratings
for user_id in user_stats.index:
    user_ratings = filtered_ratings[filtered_ratings['userId'] == user_id]
    genre_ratings = defaultdict(list)
    
    for _, rating_row in user_ratings.iterrows():
        movie_id = rating_row['movieId']
        rating = rating_row['rating']
        
        if movie_id in movie_dict:
            genres_str = movie_dict[movie_id]['genres']
            if pd.notna(genres_str) and genres_str != "(no genres listed)":
                genres = genres_str.split('|')
                for genre in genres:
                    genre_ratings[genre].append(rating)
    
    # Calculer la moyenne des ratings par genre
    for genre in unique_genres:
        if genre in genre_ratings:
            user_genre_prefs.loc[user_id, f'pref_{genre}'] = np.mean(genre_ratings[genre])


# In[20]:


user_genre_prefs.head()


# In[21]:


# ==============================
# === Stats pour les FILMS ===
# ==============================
movie_time_diffs = []

for movie_id, group in filtered_ratings.groupby("movieId"):
    group_sorted = group.sort_values("tstamp")
    if len(group_sorted) > 1:
        deltas = group_sorted["tstamp"].diff().dt.total_seconds().dropna()
        avg_delta = deltas.mean()
    else:
        avg_delta = np.nan  # pas assez de notes
    
    movie_time_diffs.append((movie_id, avg_delta))

movie_time_diffs = pd.DataFrame(movie_time_diffs, columns=["movieId", "movie_avg_time_diff"])

# Ajouter aux features films
movie_genre_features = movie_genre_features.join(movie_time_diffs.set_index("movieId"), how="left")
movie_genre_features["movie_avg_time_diff"] = movie_genre_features["movie_avg_time_diff"].fillna(0)


# In[22]:


movie_genre_features.head()


# In[23]:


# Ajouter au dictionnaire des films
for movie_id in movie_dict.keys():
    if movie_id in movie_time_diffs.set_index("movieId").index:
        movie_dict[movie_id]["avg_time_diff"] = movie_time_diffs.set_index("movieId").loc[movie_id, "movie_avg_time_diff"]
    else:
        movie_dict[movie_id]["avg_time_diff"] = 0


# In[24]:


movie_dict[1]


# In[25]:


movie_genre_features = movie_genre_features.fillna(0)
user_genre_prefs = user_genre_prefs.fillna(0)


# In[26]:


# Créer les matrices d'entraînement
training_data = []

for _, row in filtered_ratings.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    rating = row['rating']
    
    if user_id in user_stats.index and movie_id in movie_genre_features.index:
        # Features utilisateur
        user_features = [user_id, user_stats.loc[user_id, 'rating_count'], 
                       user_stats.loc[user_id, 'rating_mean'], user_stats.loc[user_id, 'user_avg_time_diff'] ]
        user_features.extend(user_genre_prefs.loc[user_id].values)
        
        # Features film
        movie_features = [movie_id]
        movie_features.extend(movie_genre_features.loc[movie_id].values)
        
        training_data.append({
            'user_features': user_features,
            'item_features': movie_features,
            'rating': rating
        })


# In[27]:


# Convertir en arrays numpy
user_train = np.array([d['user_features'] for d in training_data])
item_train = np.array([d['item_features'] for d in training_data])
y_train = np.array([d['rating'] for d in training_data])
    


# In[28]:


print("taille user train", user_train.shape)
print("taille item train", item_train.shape)
print("taille y train", y_train.shape)


# In[29]:


user_train[1]


# In[30]:


item_train[1]


# In[31]:


y_train[1]


# In[32]:


# Créer les feature names
user_feature_names = ['userId', 'rating_count', 'rating_mean'] + [f'pref_{g}' for g in unique_genres]
item_feature_names = ['movieId'] + [f'genre_{g}' for g in unique_genres]
    


# In[33]:


user_feature_names


# In[34]:


item_feature_names


# In[35]:


# Créer item_vecs (tous les films avec leurs features)
item_vecs = []
for movie_id in movie_genre_features.index:
    if movie_id in valid_movies:
        movie_features = [movie_id]
        movie_features.extend(movie_genre_features.loc[movie_id].values)
        item_vecs.append(movie_features)
item_vecs = np.array(item_vecs)


# In[36]:


item_vecs[1]


# In[37]:


# Créer user_to_genre mapping
user_to_genre = {}
for user_id in user_genre_prefs.index:
    user_to_genre[user_id] = user_genre_prefs.loc[user_id].values


# In[38]:


user_to_genre[43715]


# In[39]:


## 5. Fonctions utilitaires pour l'affichage
def gen_user_vecs(user_vec, num_items):
    """Génère des vecteurs utilisateur répétés pour tous les films"""
    return np.tile(user_vec, (num_items, 1))

def get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre):
    """Récupère les vecteurs utilisateur pour un utilisateur donné"""
    # Trouver l'utilisateur dans les données
    user_indices = np.where(user_train_unscaled[:, 0] == uid)[0]
    if len(user_indices) == 0:
        print(f"Utilisateur {uid} non trouvé")
        return None, None
    
    user_data = user_train_unscaled[user_indices[0]]
    user_vecs = gen_user_vecs(user_data.reshape(1, -1), len(item_vecs))
    
    # Créer des y_vecs factices (ratings moyens de l'utilisateur)
    y_vecs = np.full(len(item_vecs), user_data[2])
    
    return user_vecs, y_vecs

def print_pred_movies(y_p, item_vecs, movie_dict, maxcount=10):
    """Affiche les prédictions de films"""
    print(f"\nTop {maxcount} film recommendations:")
    print("Rating | Title")
    print("-" * 50)
    
    for i in range(min(maxcount, len(y_p))):
        movie_id = int(item_vecs[i][0])
        if movie_id in movie_dict:
            print(f"{y_p[i][0]:4.1f}   | {movie_dict[movie_id]['title']}")

def print_existing_user(y_p, y_actual, user_vecs, item_vecs, ivs, uvs, movie_dict, maxcount=50):
    """Affiche les prédictions pour un utilisateur existant"""
    print(f"\nPrédictions vs ratings réels (top {maxcount}):")
    print("Pred | Actual | Title")
    print("-" * 60)
    
    for i in range(min(maxcount, len(y_p))):
        movie_id = int(item_vecs[i][0])
        if movie_id in movie_dict:
            print(f"{y_p[i][0]:4.1f} | {y_actual[i][0]:4.1f}   | {movie_dict[movie_id]['title']}")


# In[40]:


## 6. Normalisation et division des données

# Scale training data
item_train_unscaled = item_train.copy()
user_train_unscaled = user_train.copy()
y_train_unscaled = y_train.copy()

# Normalisation des features des films
scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

# Normalisation des features des utilisateurs 
scalerUser = StandardScaler() 
scalerUser.fit(user_train) 
user_train = scalerUser.transform(user_train)

# Normalisation des targets (ratings) 
scalerTarget = MinMaxScaler((-1, 1)) 
scalerTarget.fit(y_train.reshape(-1, 1)) 
y_train = scalerTarget.transform(y_train.reshape(-1, 1))

print("Vérification de l'inverse transform:")
print(f"Items: {np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train))}")
print(f"Users: {np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train))}")


# In[41]:


### Division train/test

# Split des données
item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test = train_test_split(y_train, train_size=0.80, shuffle=True, random_state=1)

print(f"Données d'entraînement:")
print(f"- Films: {item_train.shape}")
print(f"- Utilisateurs: {user_train.shape}")
print(f"- Ratings: {y_train.shape}")
print(f"\nDonnées de test:")
print(f"- Films: {item_test.shape}")
print(f"- Utilisateurs: {user_test.shape}")
print(f"- Ratings: {y_test.shape}")


# In[ ]:





# In[60]:


num_user_features = user_train[0].shape
num_item_features = item_train[0].shape


# In[61]:


## 7. Construction du modèle de recommandation
from keras.layers import Lambda

num_outputs = 32
tf.random.set_seed(1)
user_NN = keras.models.Sequential([
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(num_outputs),
  ])

item_NN = keras.models.Sequential([
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(num_outputs),
  
])

# create the user input and point to the base network
input_user = keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)

# create the item input and point to the base network
input_item = keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)

# compute the dot product of the two vectors vu and vm
output = keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = keras.Model([input_user, input_item], output)

model.summary()


# In[62]:


## 8. Entraînement du modèle

### Configuration de l'optimiseur

tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

print("Modèle compilé avec:")
print(f"- Fonction de coût: {cost_fn.__class__.__name__}")
print(f"- Optimiseur: {opt.__class__.__name__}")
print(f"- Learning rate: {opt.learning_rate.numpy()}")


# In[63]:


user_train[0]


# In[64]:


item_train[0]


# In[ ]:


### Lancement de l'entraînement

("🚀 Début de l'entraînement...")
tf.random.set_seed(1)

history = model.fit(
    [
        user_train,   # ou bien le vrai nom de ton Input utilisateur
        item_train    # idem pour le film
    ],
    y_train,
    epochs=30,
    verbose=1,
    validation_split=0.1
)

("✅ Entraînement terminé!")


# In[ ]:


### Évaluation sur les données de test


# Évaluation du modèle
test_loss = model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)
print(f"\n📊 Performance du modèle:")
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"RMSE: {np.sqrt(test_loss):.4f}")


# In[ ]:


### Visualisation de l'entraînement

import matplotlib.pyplot as plt

# Graphique des pertes d'entraînement
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Evolution de la perte')
plt.xlabel('Époque')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:


## 9. Recommandations pour un nouvel utilisateur


print("🎬 RECOMMANDATIONS POUR UN NOUVEL UTILISATEUR")
print("=" * 60)

# Créer un profil de nouvel utilisateur (amateur d'aventure et de fantasy)
new_user_id = 999999
new_rating_ave = 3.5
new_rating_count = 10

# Créer le vecteur utilisateur (préférences par genre)
user_prefs = {}
for genre in genres:
    if genre in ['Adventure', 'Fantasy', 'Action']:
        user_prefs[f'pref_{genre}'] = 4.5  # Forte préférence
    elif genre in ['Romance', 'Comedy']:
        user_prefs[f'pref_{genre}'] = 2.0  # Faible préférence
    else:
        user_prefs[f'pref_{genre}'] = 3.0  # Préférence neutre

print("Profil utilisateur créé:")
print(f"- Rating moyen: {new_rating_ave}")
print(f"- Nombre de ratings: {new_rating_count}")
print("- Préférences fortes: Adventure, Fantasy, Action")
print("- Préférences faibles: Romance, Comedy")

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave] + 
                    [user_prefs[f'pref_{genre}'] for genre in genres]])

# Générer les vecteurs utilisateur pour tous les films
user_vecs = gen_user_vecs(user_vec, len(item_vecs))

# Normaliser les données
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# Faire les prédictions
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# Dénormaliser les prédictions
y_pu = scalerTarget.inverse_transform(y_p)

# Trier les résultats
sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()
sorted_ypu = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]

print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount=15)



# In[ ]:


## 10. Recommandations pour un utilisateur existant

print("👤 RECOMMANDATIONS POUR UN UTILISATEUR EXISTANT")
print("=" * 60)

# Choisir un utilisateur existant
uid = int(user_train_unscaled[100, 0])  # Prendre le 100e utilisateur
print(f"Analyse des recommandations pour l'utilisateur {uid}")

# Obtenir les vecteurs utilisateur
user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)

if user_vecs is not None:
    # Afficher le profil de l'utilisateur
    user_profile = user_train_unscaled[user_train_unscaled[:, 0] == uid][0]
    print(f"Profil utilisateur {uid}:")
    print(f"- Rating moyen: {user_profile[2]:.2f}")
    print(f"- Nombre de ratings: {int(user_profile[1])}")
    
    # Normaliser les données
    suser_vecs = scalerUser.transform(user_vecs)
    sitem_vecs = scalerItem.transform(item_vecs)
    
    # Faire les prédictions
    y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])
    
    # Dénormaliser les prédictions
    y_pu = scalerTarget.inverse_transform(y_p)
    
    # Trier les résultats
    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()
    sorted_ypu = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user = user_vecs[sorted_index]
    sorted_y = y_vecs[sorted_index]
    
    print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, 
                       sorted_items, ivs, uvs, movie_dict, maxcount=25)


# In[ ]:


## 11. Analyse de similarité des films

print("🎭 ANALYSE DE SIMILARITÉ DES FILMS")
print("=" * 60)

def sq_dist(a, b):
    """Calcule la distance euclidienne au carré entre deux vecteurs"""
    return np.sum((b - a) ** 2)

# Test de la fonction de distance
a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
a3 = np.array([0, 1, 0]); b3 = np.array([1, 0, 0])

print("Tests de la fonction de distance:")
print(f"Distance entre vecteurs identiques: {sq_dist(a1, b1):.3f}")
print(f"Distance entre vecteurs similaires: {sq_dist(a2, b2):.3f}")
print(f"Distance entre vecteurs différents: {sq_dist(a3, b3):.3f}")


# In[ ]:


### Extraction des embeddings des films

# Créer un modèle pour extraire les embeddings des films
input_item_m = tf.keras.layers.Input(shape=(num_item_features,))
vm_m = item_NN(input_item_m)
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)
model_m = tf.keras.Model(input_item_m, vm_m)
model_m.summary()

# Obtenir les embeddings de tous les films
scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:, i_s:])
print(f"Taille des vecteurs de features des films: {vms.shape}")


# In[ ]:


### Calcul et affichage des similarités

# Calculer les distances entre films
count = 100  # nombre de films à analyser pour les similarités
dim = len(vms)
dist = np.zeros((dim, dim))

print("Calcul des distances entre films...")
for i in range(min(count, dim)):
    for j in range(dim):
        dist[i, j] = sq_dist(vms[i, :], vms[j, :])

# Masquer la diagonale
m_dist = ma.masked_array(dist[:count], 
                        mask=np.identity(count) if count <= dim else np.zeros((count, dim)))

# Afficher les films les plus similaires
print(f"\nFilms les plus similaires (analyse sur {min(count, dim)} films):")
print("=" * 100)
print(f"{'Film Original':<40} {'Genres':<25} {'Film Similaire':<40} {'Genres'}")
print("=" * 100)

similar_pairs = []
for i in range(min(count, 25)):  # Limiter l'affichage à 25 paires
    if i < len(m_dist):
        min_idx = np.argmin(m_dist[i])
        movie1_id = int(item_vecs[i, 0])
        movie2_id = int(item_vecs[min_idx, 0])
        
        if movie1_id in movie_dict and movie2_id in movie_dict:
            movie1_title = movie_dict[movie1_id]['title']
            movie2_title = movie_dict[movie2_id]['title']
            movie1_genres = movie_dict[movie1_id]['genres']
            movie2_genres = movie_dict[movie2_id]['genres']
            
            # Tronquer les titres si trop longs
            movie1_display = movie1_title[:37] + "..." if len(movie1_title) > 40 else movie1_title
            movie2_display = movie2_title[:37] + "..." if len(movie2_title) > 40 else movie2_title
            movie1_genres_display = movie1_genres[:22] + "..." if len(movie1_genres) > 25 else movie1_genres
            movie2_genres_display = movie2_genres[:22] + "..." if len(movie2_genres) > 25 else movie2_genres
            
            print(f"{movie1_display:<40} {movie1_genres_display:<25} {movie2_display:<40} {movie2_genres_display}")
            
            similar_pairs.append({
                'movie1': movie1_title,
                'movie2': movie2_title,
                'genres1': movie1_genres,
                'genres2': movie2_genres,
                'distance': dist[i, min_idx]
            })

print("=" * 100)


# In[ ]:


## 12. Analyse des résultats et métriques

print("📈 ANALYSE DES RÉSULTATS")
print("=" * 60)

# Statistiques sur les données
print("Statistiques des données d'entraînement:")
print(f"- Nombre total de ratings: {len(y_train_unscaled):,}")
print(f"- Rating moyen: {np.mean(y_train_unscaled):.2f}")
print(f"- Écart-type des ratings: {np.std(y_train_unscaled):.2f}")
print(f"- Rating minimum: {np.min(y_train_unscaled):.1f}")
print(f"- Rating maximum: {np.max(y_train_unscaled):.1f}")

# Distribution des genres
print(f"\nStatistiques des genres ({len(genres)} genres uniques):")
genre_counts = {}
for movie_id, movie_info in movie_dict.items():
    if movie_info['genres'] and movie_info['genres'] != "(no genres listed)":
        for genre in movie_info['genres'].split('|'):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 des genres les plus fréquents:")
for genre, count in top_genres:
    print(f"- {genre}: {count:,} films")


# In[ ]:


### Évaluation de la qualité des recommandations

# Calculer quelques métriques de qualité
def evaluate_recommendations(model, user_test, item_test, y_test, scalers, num_samples=1000):
    """Évalue la qualité des recommandations"""
    
    # Prendre un échantillon pour l'évaluation
    indices = np.random.choice(len(user_test), min(num_samples, len(user_test)), replace=False)
    
    user_sample = user_test[indices]
    item_sample = item_test[indices]
    y_sample = y_test[indices]
    
    # Prédictions
    y_pred = model.predict([user_sample[:, u_s:], item_sample[:, i_s:]])
    
    # Dénormaliser
    y_true_denorm = scalers['target'].inverse_transform(y_sample)
    y_pred_denorm = scalers['target'].inverse_transform(y_pred)
    
    # Calculer les métriques
    mse = np.mean


    

