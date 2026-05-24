import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroDQN(nn.Module):
    """
    Tactical 15x15 combat network.
    Input Layer: 7 Neurons (Internal State: hp_percent, level; Target Data: delta_x, delta_y, target_hp_percent; Spatial: distance_to_nearest_edge; Game State: Flee_Attempts_Remaining)
    Hidden Layers: 64 -> 64
    Output Layer: 9 Neurons (Raw Q-values for: 8 directional movements, BASIC_ATTACK)
    """
    def __init__(self):
        super(MicroDQN, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
