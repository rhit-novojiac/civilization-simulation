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

# Pre-allocated tensors to prevent memory allocation overhead in the hot loop
_MAX_BATCH = 20000
_biomass_mask = torch.zeros(_MAX_BATCH, dtype=torch.bool)
_diet_mask = torch.zeros(_MAX_BATCH, dtype=torch.bool)

def select_macro_actions(q_move, q_stance, epsilon, biomass_list=None, threshold=100.0, diet_list=None):
    """
    Takes Q-value tensors from the dual-headed MacroDQN and returns 
    a list of (move_action, stance_action) tuples using epsilon-greedy selection.
    If biomass_list is provided, masks out ESTABLISH_DEN (5) for entities with biomass < threshold.
    """
    batch_size = q_move.size(0)
    
    # Action Masking (Vectorized with pre-allocated tensors)
    if biomass_list is not None:
        for i in range(batch_size):
            _biomass_mask[i] = (biomass_list[i] < threshold)
        q_move[_biomass_mask[:batch_size], MacroAction.ESTABLISH_DEN] = -float('inf')
                
    if diet_list is not None:
        for i in range(batch_size):
            _diet_mask[i] = (diet_list[i] == "Herbivore")
        q_stance[_diet_mask[:batch_size], MacroStance.AGGRESSIVE] = -float('inf')
                
    argmax_move = q_move.argmax(dim=1).tolist()
    argmax_stance = q_stance.argmax(dim=1).tolist()
    
    actions = []
    for i in range(batch_size):
        # 1. The Override Logic: Biological Imperative
        # if biomass_list is not None and biomass_list[i] >= threshold:
        #     m_act = int(MacroAction.ESTABLISH_DEN)
        #     s_act = 1 if diet_list is not None and diet_list[i] == "Herbivore" else argmax_stance[i]
        #     actions.append((m_act, s_act))
        #     continue
            
        if random.random() < epsilon:
            # Random selection with masking
            if biomass_list is not None and biomass_list[i] < threshold:
                valid_actions = [0, 1, 2, 3, 4]
                m_act = random.choice(valid_actions)
            else:
                valid_actions = [0, 1, 2, 3, 4, 5]
                m_act = random.choice(valid_actions)
                
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
    BASIC_ATTACK = 0
    FLEE = 1
    TOLERATE = 2
