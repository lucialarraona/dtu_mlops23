import argparse
import sys

#import multiprocessing
#multiprocessing.set_start_method('spawn')

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import click

from torch import nn
from torch import optim

from data import mnist
from data import toDataLoader
from model import MyAwesomeModel



@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')

def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, test_set = mnist() #import data
    train_loader, _ = toDataLoader(train_set,test_set)
   

    # Hyperparameters 
    loss_func = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr = 0.01)   
    num_epochs = 10


    # --------------------- Train loop -----------------   
    
    for epoch in range(num_epochs): 

        model.train()
        total_step = len(train_loader)
            
        for i, (images, labels) in enumerate(train_loader):  # dataloaders for faster computation 
            #image = image.to('cuda') 
            #label = label.to('cuda') 

            output = model(images)[0] 

            loss = loss_func(output, labels)
            # clear gradients for this training step   
            optimizer.zero_grad()           
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()   

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
        
        pass
    
    
    pass
            
    


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    train_set, test_set = mnist()
    _, test_loader = toDataLoader(train_set,test_set)

    # Evaluation Loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    
    pass





cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    