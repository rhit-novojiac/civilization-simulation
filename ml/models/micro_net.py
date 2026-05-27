import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroDQN(nn.Module):
    """
    Abstract Auto-Battler tactical network.
    Input Layer: 12 Neurons (Tier Injected Stats)
    Hidden Layers: 64 -> 64
    Output Layer: 3 Neurons [BASIC_ATTACK, FLEE, TOLERATE]
    """
    def __init__(self):
        super(MicroDQN, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
