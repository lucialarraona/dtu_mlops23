import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import transforms
import helper 

class MNISTdata(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x.float(), y

    def __len__(self):
        return len(self.data)


def mnist():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_paths = [f"/Users/lucialarraona/Desktop/dtu_mlops23/data/corruptmnist/train_{i}.npz" for i in range(5)]

    X_train = np.concatenate(
        [np.load(train_file)["images"] for train_file in train_paths])
    Y_train = np.concatenate(
        [np.load(train_file)["labels"] for train_file in train_paths])

    X_test = np.load("/Users/lucialarraona/Desktop/dtu_mlops23/data/corruptmnist/test.npz")["images"]
    Y_test = np.load("/Users/lucialarraona/Desktop/dtu_mlops23/data/corruptmnist/test.npz")["labels"]

    train = MNISTdata(X_train, Y_train, transform=transform)
    #trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = MNISTdata(X_test, Y_test, transform=transform)
    #testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    return train, test


# Sanity check - Visualize data
train, test = mnist()
print(len(train))


# Multiple train data
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train), size=(1,)).item()
    img = train[sample_idx][0]
    label = train[sample_idx][1]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = 'gray') 
plt.show()


