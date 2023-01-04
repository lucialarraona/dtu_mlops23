import argparse
import sys

import click
#import matplotlib.pyplot as plt
import torch
import model 
from model import MyAwesomeModel
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

#from data import mnist
from src.data.make_dataset import MNISTdata
import logging 

log = logging.getLogger(__name__)

def main():
    device = "cpu"
    mymodel = MyAwesomeModel()
    mymodel = mymodel.to(device)

    # Access data from processed folder
    train_data = torch.load("data/processed/train.pth")
    test_data = torch.load("data/processed/test.pth")

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=True)

    # hparams
 
    optimizer = torch.optim.Adam(mymodel.parameters(), lr = 1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    log.info("Start training...")

    # modified the training function so that it takes  hyperparameters
    model.train(
        mymodel,
        trainloader,
        criterion,
        optimizer,
        epochs = 5)

    # Save thecheckpoint
    checkpoint = {
    "hidden_size": 128,
    "output_size": 10,
    "state_dict": mymodel.state_dict()}

    log.info("Finish! :D")

    # Save checkpoint in the models
    torch.save(checkpoint, "models/checkpoint.pth")



if __name__ == "__main__":
    main()
