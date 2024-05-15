import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,3,5,1,1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(3,16,3,1,1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.Flatten(),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )


    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x