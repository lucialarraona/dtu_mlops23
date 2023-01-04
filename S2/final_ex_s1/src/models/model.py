import torch.nn.functional as F
from torch import nn
import matplotlib as plt
import torch 

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


def train(model,trainloader, loss_func, optimizer, epochs, lr):
    print("Training day and night")
    print(lr)


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
        
    plt.plot(loss_tracker, '-')
    plt.xlabel('Training step')
    plt.ylabel('Training loss')
    plt.savefig("/Users/lucialarraona/Desktop/dtu_mlops23/S2/final_ex_s1/reports/figures/training_curve.png")
    
    return model



def evaluate(model_checkpoint,model, testloader,loss_func,):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    device = 'cpu'
    model.load_state_dict(torch.load(model.checkpoint))
    model = model.to(device)
    
    correct, total = 0, 0
    for batch in testloader:
        x, y = batch
        
        preds = model(x.to(device))
        preds = preds.argmax(dim=-1)
        
        correct += (preds == y.to(device)).sum().item()
        total += y.numel()
        
    print(f"Test set accuracy {correct/total}")

