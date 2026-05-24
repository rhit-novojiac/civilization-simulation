import random
import torch
from enum import IntEnum

class MacroAction(IntEnum):
    MOVE_N = 0
    MOVE_S = 1
    MOVE_E = 2
    MOVE_W = 3
    REST = 4
    ESTABLISH_DEN = 5

class MacroStance(IntEnum):
    AGGRESSIVE = 0
    PEACEFUL = 1

def select_macro_actions(q_move, q_stance, epsilon, biomass_list=None, threshold=100.0, diet_list=None):
    """
    Takes Q-value tensors from the dual-headed MacroDQN and returns 
    a list of (move_action, stance_action) tuples using epsilon-greedy selection.
    If biomass_list is provided, masks out ESTABLISH_DEN (5) for entities with biomass < threshold.
    """
    batch_size = q_move.size(0)
    
    # Action Masking
    if biomass_list is not None:
        for i in range(batch_size):
            if biomass_list[i] < threshold:
                q_move[i, MacroAction.ESTABLISH_DEN] = -99999.0
                
    if diet_list is not None:
        for i in range(batch_size):
            if diet_list[i] == "Herbivore":
                q_stance[i, MacroStance.AGGRESSIVE] = -99999.0
                
    argmax_move = q_move.argmax(dim=1).tolist()
    argmax_stance = q_stance.argmax(dim=1).tolist()
    
    actions = []
    for i in range(batch_size):
        if random.random() < epsilon:
            # Random selection with masking
            if biomass_list is not None and biomass_list[i] < threshold:
                m_act = random.randint(0, 4)
            else:
                m_act = random.randint(0, 5)
                
            if diet_list is not None and diet_list[i] == "Herbivore":
                s_act = 1 # PEACEFUL
            else:
                s_act = random.randint(0, 1)
        else:
            m_act = argmax_move[i]
            s_act = argmax_stance[i]
        actions.append((m_act, s_act))
        
    return actions

class MicroAction(IntEnum):
    MOVE_N = 0
    MOVE_S = 1
    MOVE_E = 2
    MOVE_W = 3
    MOVE_NE = 4
    MOVE_NW = 5
    MOVE_SE = 6
    MOVE_SW = 7
    BASIC_ATTACK = 8
