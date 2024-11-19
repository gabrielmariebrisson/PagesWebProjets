from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import de pad_sequences

app = Flask(__name__, static_folder='templates/assets', static_url_path='/assets')

# Charger le modèle
loaded_model = tf.keras.models.load_model('./templates/assets/AnalyseSentiment.h5')

def seq_pad_and_trunc(sequences, tokenizer, padding='post', truncating='post', maxlen=100):
    sequences = tokenizer.texts_to_sequences([sequences])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
    return padded_sequences

@app.route('/AnalyseSentiment', methods=['GET', 'POST'])
def index():
    prediction = None  # Initialise la variable pour stocker la prédiction
    if request.method == 'POST':
        # Obtenir le texte de la requête
        input_text = request.form['text']

        with open('./templates/assets/tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Prétraitement du texte
        test_pad_trunc_seq = seq_pad_and_trunc(input_text, tokenizer)

        # Faire la prédiction
        predictions = loaded_model.predict(test_pad_trunc_seq)

        # Convertir les probabilités en classes (0 ou 1)
        predicted_class = (predictions > 0.5).astype(int)[0][0]
        prediction = {'prediction': str(predicted_class)}

    return render_template('AnalyseSentiment.html', prediction=prediction)

def redirect_to_other_site():
    return redirect("https://gabriel.mariebrisson.fr", code=301)

if __name__ == '__main__':
    app.run(debug=True)
