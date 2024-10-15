import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
from TripleMNIST import TripleMNIST
from Trainer import Trainer
from CNN import CNN


root = "/Users/karolinanowacka/Desktop/ML projects/SOLVRO/intro_task/MNIST_tripplet_loss"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        

dataset = TripleMNIST()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
data_iter = iter(dataloader)
anchor_imgs, positive_imgs, negative_imgs = next(data_iter)


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.TripletMarginLoss(margin = 1.0, p = 2)

print("initializing trainer...")
trainer = Trainer(model, dataloader, optimizer, loss_fn, epochs = 1)

print("training...")
trainer.train()