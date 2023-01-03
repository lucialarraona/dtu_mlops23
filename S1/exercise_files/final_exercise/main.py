import argparse
import sys

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import click

from torch import nn
from torch import optim

from data import mnist
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
    
    device = 'cpu'
    model = MyAwesomeModel()
    model = model.to(device)
    #train_set = CorruptMnist(train=True)
    train_set, test_set= mnist() #import data
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    n_epoch = 5
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x.to(device))
            loss = criterion(preds, y.to(device))
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")        
    torch.save(model.state_dict(), 'trained_model.pt')
        
    plt.plot(loss_tracker, '-')
    plt.xlabel('Training step')
    plt.ylabel('Training loss')
    plt.savefig("training_curve.png")
    
    return model



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    device = 'cpu'
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model.checkpoint))
    model = model.to(device)

    #test_set = CorruptMnist(train=False)
    train_set, test_set= mnist() #import data
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
    
    correct, total = 0, 0
    for batch in dataloader:
        x, y = batch
        
        preds = model(x.to(device))
        preds = preds.argmax(dim=-1)
        
        correct += (preds == y.to(device)).sum().item()
        total += y.numel()
        
    print(f"Test set accuracy {correct/total}")



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    