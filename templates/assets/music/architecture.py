import torch
import torch.nn as nn
import torch.nn.functional as F

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max_pool.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg_pool.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max_pool = torch.max(x,1)[0].unsqueeze(1)
        avg_pool = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max_pool,avg_pool), dim=1)
        output = self.conv(concat)
        output = torch.sigmoid(output) * x 
        return output     

class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super(SimpleCNN, self).__init__()
        
        self.norm1 = nn.BatchNorm2d(1) 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(128,4), padding=(0,1))
        self.relu = nn.ReLU()
        self.cbam1 = CBAM(256, r=16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,4))
        
        self.norm2 = nn.BatchNorm2d(1) 
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(256,4)) 
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2))
        
        self.norm3 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(256,4))
        self.cbam2 = CBAM(512, r=16)
        
        self.norm4 = nn.BatchNorm2d(1) 
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(512,4))
        
        self.normfc2 = nn.BatchNorm1d(1536)
        self.fc2 = nn.Linear(1536, 2048)
        self.normfc3 = nn.BatchNorm1d(2048)
        self.drop1 = nn.Dropout2d(p=dropout)  
        self.fc3 = nn.Linear(2048, 1024)
        self.normfc4 = nn.BatchNorm1d(1024) 
        self.drop2 = nn.Dropout2d(p=dropout) 
        self.fc4 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.cbam1(x)
        x = self.pool1(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.cbam1(x)
        x = self.pool2(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = self.norm3(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.cbam2(x)
        x = self.pool2(x)
        x = torch.permute(x,(0,2,1,3))
        
        x = self.norm4(x)
        x = self.conv4(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = torch.permute(x,(0,2,1,3))
        
        mean_values = torch.mean(x, dim=3, keepdim=True)
        max_values, _ = torch.max(x, dim=3, keepdim=True)
        l2_norm = torch.linalg.norm(x, dim=3, ord=2, keepdim=True)
        
        x = torch.cat([max_values, mean_values, l2_norm], dim=1)
        x = x.view(-1, 1536)
        
        x = self.normfc2(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.normfc3(x)
        x = self.fc3(x)
        x = self.drop1(x)
        x = F.relu(x)
        
        x = self.normfc4(x)
        x = self.drop2(x)
        x = self.fc4(x)
        
        return x