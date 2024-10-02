import torch as torch
import matplotlib.pyplot as plt
from torchvision import datasets as dt
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


root = "/Users/karolinanowacka/Desktop/ML projects/SOLVRO/intro_task/MNIST_tripplet_loss"

train_data = dt.MNIST(
    root=root, 
    train=True, 
    download=True,
    transform=ToTensor()
    )

test_data = dt.MNIST(
    root=root, 
    train=False, 
    download=True,
    transform=ToTensor()
    )

labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}


figure = plt.figure()
cols=3
rows=3

for i in range(1, cols*rows +1):
    sample_idx = torch.randint(len(train_data)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

