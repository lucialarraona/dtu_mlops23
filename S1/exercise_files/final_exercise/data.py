import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import transforms
import helper 


def mnist():
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784) 

    train = np.load('/Users/lucialarraona/dtu_mlops/data/corruptmnist/train_0.npz', allow_pickle=True)
    test = np.load('/Users/lucialarraona/dtu_mlops/data/corruptmnist/test.npz', allow_pickle=True)
    
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])  
    
    # How do I apply the transformations to the images?
    train = list(zip(train['images'], train['labels']))
    test = list(zip(test['images'], test['labels']))

    #Convert to dataloader for training
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

    return train_loader, test_loader



# Sanity check - Visualize data
train_loader, test_loader = mnist()
print(len(train_loader))



image, label = next(iter(train_loader))
helper.imshow(image[0,:]);

"""
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
"""

