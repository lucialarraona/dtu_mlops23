import argparse
import sys

import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import mnist


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here

    device = "cpu"
    model = MyAwesomeModel()
    model = model.to(device)
    # train_set = CorruptMnist(train=True)
    train_set, test_set = mnist()  # import data
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
    torch.save(
        model.state_dict(),
        "/Users/lucialarraona/Desktop/dtu_mlops23/S2/final_ex_s1/models/trained_model.pt",
    )  # save trained model in the models folder

    plt.plot(loss_tracker, "-")
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.savefig(
        "/Users/lucialarraona/Desktop/dtu_mlops23/S2/final_ex_s1/reports/figures/training_curve.png"
    )  # Save plot in the figures folder

    return model


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    device = "cpu"
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model.checkpoint))
    model = model.to(device)

    # test_set = CorruptMnist(train=False)
    train_set, test_set = mnist()  # import data
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
