from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np


root = "/Users/karolinanowacka/Desktop/ML projects/SOLVRO/intro_task/MNIST_tripplet_loss"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class TripleMNIST(Dataset):
    def __init__(self):
        super().__init__()
        self.MNIST = datasets.MNIST(root = root, train = True, download = True, transform = transform)
        self.grouped_by_label = {i: [] for i in range(10)}
        
        for i in range(len(self.MNIST)):
            label = self.MNIST[i][1]
            self.grouped_by_label[label].append(i)

    def get_grouped_by_label(self):
        return self.grouped_by_label

    def __len__(self):
        return len(self.MNIST)

    def __getitem__(self, anchor_idx):
        anchor_img, anchor_label = self.MNIST[anchor_idx]
        
        positive_idx = np.random.choice(self.grouped_by_label[anchor_label])
        positive_img, _ = self.MNIST[positive_idx]
        
        negative_label = np.random.choice([label for label in range(10) if label != anchor_label])
        negative_idx = np.random.choice(self.grouped_by_label[negative_label])
        negative_img, _ = self.MNIST[negative_idx]
        
        return anchor_img, positive_img, negative_img