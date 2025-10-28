# model.py
import torch.nn as nn
import torch

class CustomNetwork(nn.Module):
    """
    MLP for Fashion-MNIST:
    784 -> 300 -> 200 -> 10 with Dropout and ReLU.
    Returns raw logits (use CrossEntropyLoss).
    """
    def __init__(self, in_dim: int = 28*28, n_hidden1: int = 300, n_hidden2: int = 200,
                 n_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, n_hidden1)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.do2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(n_hidden2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28) or (B, 784)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.do1(x)
        x = torch.relu(self.fc2(x))
        x = self.do2(x)
        x = self.fc3(x)  # logits
        return x
