import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin = 1.0, p = 2)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1) 
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 10)          
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(-1, 64 * 7 * 7) 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    