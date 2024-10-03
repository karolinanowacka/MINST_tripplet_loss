import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#TODO: rozwayć batch normalization, użyć w modelu warstw liniowych do redukcji wymiarowości

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) # 1 input channel, 6 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d()

        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)


