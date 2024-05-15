import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, num_class):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x