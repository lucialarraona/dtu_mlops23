import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torch 
import hydra

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),  # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),  # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),  # [N, 8, 20]
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, 128), nn.Dropout(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# Sanity check (print model architecture)
model = MyAwesomeModel()
print(model)


def train(model,trainloader, loss_func, optimizer, epochs):
    print("Training day and night")
    #print(lr)


    # TODO: Implement training loop here
    
    device = 'cpu'
    model = model.to(device)

    n_epoch = epochs
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in trainloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x.to(device))
            loss = loss_func(preds, y.to(device))
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")        
   # torch.save(model.state_dict(), 'trained_model.pt')
        
    #plt.plot(loss_tracker, '-')
    #plt.xlabel('Training step')
    #plt.ylabel('Training loss')
    #plt.savefig("reports/figures/training_curve.png")
    
    return model



def evaluate(model, testloader,criterion):
    print("Evaluating until hitting the ceiling")
    print(model)

    # TODO: Implement evaluation logic here
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    #print(f"Test set accuracy {test_loss}")

    return test_loss, accuracy, ps.max(1)[1], labels.data, images

        
  

