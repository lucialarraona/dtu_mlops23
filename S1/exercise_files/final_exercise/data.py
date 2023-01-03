import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import transforms
import helper 


def mnist1():
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784) 

    train_raw = np.load('/Users/lucialarraona/dtu_mlops/data/corruptmnist/train_0.npz', allow_pickle=True)
    test_raw = np.load('/Users/lucialarraona/dtu_mlops/data/corruptmnist/test.npz', allow_pickle=True)
    
    #train_img_reshape = torch.tensor(np.concatenate([c['images'] for c in train_raw])).reshape(-1, 1, 28, 28)
    #test_img_reshape = torch.tensor(np.concatenate([c['images'] for c in test_raw])).reshape(-1, 1, 28, 28)
    
    # Define a transform to normalize the data
    #transform = transforms.Compose([transforms.ToTensor(),
    #                            transforms.Normalize((0.5,), (0.5,))])  
    
    # How do I apply the transformations to the images?
    train = list(zip(train_raw['images'].reshape(-1, 1, 28, 28).astype(np.float32), train_raw['labels']))
    test = list(zip(test_raw['images'].reshape(-1, 1, 28, 28).astype(np.float32), test_raw['labels']))

    #Convert to dataloader for training
    #train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

    return train, test

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
    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = MNISTdata(X_test, Y_test, transform=transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    return train, test



# Sanity check - Visualize data
train, test = mnist()
print(len(train))



#image, label = next(iter(train_loader))
#print(image)
#print(label)
#helper.imshow(image[0,:]);


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


