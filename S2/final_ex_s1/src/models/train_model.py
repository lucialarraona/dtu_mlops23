import argparse
import sys
import os 

import click
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import torch
import model 
from model import MyAwesomeModel
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

#from data import mnist
from src.data.make_dataset import MNISTdata
import logging 
HYDRA_FULL_ERROR=1 

log = logging.getLogger(__name__)

# Hydra hyperparameter configuration

@hydra.main(
    config_name="training_conf.yml", config_path="config")  # hydra currently supports only 1 config file


def main(cfg):

    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    os.chdir(hydra.utils.get_original_cwd())
    hparams = cfg.experiment
    torch.manual_seed(hparams["seed"])

    device = "cpu"
    mymodel = MyAwesomeModel()
    mymodel = mymodel.to(device)

    # Access data from processed folder
    train_data = torch.load("data/processed/train.pth")
    test_data = torch.load("data/processed/test.pth")

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=hparams["batch_size"], shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=hparams["batch_size"], shuffle=True)

    # hparams
    optimizer = torch.optim.Adam(mymodel.parameters(), lr = hparams["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    log.info("Start training...")

    # modified the training function so that it takes  hyperparameters
    model.train(
        mymodel,
        trainloader,
        criterion,
        optimizer,
        epochs = hparams["n_epochs"])

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
