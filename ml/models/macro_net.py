import torch
import torch.nn as nn
import torch.nn.functional as F

class MacroDQN(nn.Module):
    """
    Overworld 7x7 vision grid network.
    Input Layer: 152 Neurons (49 tiles * 3 features + 3 internal states + 2 scent compass vectors)
    Hidden Layers: 128 -> 128
    Output Layer: 6 Neurons (Raw Q-values for: MOVE_N, MOVE_S, MOVE_E, MOVE_W, REST, ESTABLISH_DEN)
    """
    def __init__(self):
        super(MacroDQN, self).__init__()
        self.fc1 = nn.Linear(152, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out_move = nn.Linear(128, 6)
        self.out_stance = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out_move(x), self.out_stance(x)
