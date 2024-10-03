import torch as torch
import matplotlib.pyplot as plt
from torchvision import datasets as dt
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

root = "/Users/karolinanowacka/Desktop/ML projects/SOLVRO/intro_task/MNIST_tripplet_loss"

train_data = dt.MNIST(
    root=root, 
    train=True, 
    download=True,
    transform=ToTensor()
    )
#debugging
print(len(train_data))
print(train_data[0])

test_data = dt.MNIST(
    root=root, 
    train=False, 
    download=True,
    transform=ToTensor()
    )
#debugging
print(len(test_data))

#normalization


train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)

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

figure = plt.figure(figsize=(8,8))
cols=3
rows=3

for i in range(1, cols*rows +1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

train_features, train_labels = next(iter(train_data_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")


