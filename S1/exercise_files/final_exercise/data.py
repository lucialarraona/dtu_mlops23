import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import transforms



def mnist():
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784) 

    train = np.load('/Users/lucialarraona/dtu_mlops/data/corruptmnist/train_0.npz', allow_pickle=True)
    test = np.load('/Users/lucialarraona/dtu_mlops/data/corruptmnist/test.npz', allow_pickle=True)
    

    train = list(zip(torch.tensor(np.float32(train['images'])), torch.tensor(np.float32(train['labels']))))
    test = list(zip(torch.tensor(np.float32(test['images'])), torch.tensor(np.float32(test['labels']))))


    return train, test



# Sanity check - Visualize data
train, test = mnist()

#print(list(train.keys()))
print(len(train))
#print(train)

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
    plt.imshow(img.squeeze()) #cmap = 'grayscale'
plt.show()


# Transform to dataloaders
def toDataLoader(train,test):
    """
    Converts to dataloaders for better training optimization
    """
    train_loader = torch.utils.data.DataLoader(train, 
                                          batch_size=128, 
                                          shuffle=True
                                          #num_workers=1
                                          )
    
    test_loader = torch.utils.data.DataLoader(test, 
                                          batch_size=128, 
                                          shuffle=True
                                          #num_workers=1
                                          )
    return train_loader, test_loader


train_loader, test_loader = toDataLoader(train,test)