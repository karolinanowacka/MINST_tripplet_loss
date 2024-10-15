from torch.utils.data import DataLoader
from triple_mnist import TripleMNIST
import matplotlib.pyplot as plt
import numpy as np

dataset = TripleMNIST()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
data_iter = iter(dataloader)
anchor_imgs, positive_imgs, negative_imgs = next(data_iter)


def imshow(img, title):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap = 'gray')
    plt.title(title)
    plt.show()

for i in range(4):
    imshow(anchor_imgs[i], title = "anchor")
    imshow(positive_imgs[i], title = "positive")
    imshow(negative_imgs[i], title = "negative")