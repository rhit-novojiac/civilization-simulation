import math

def get_macro_reward(
    current_biomass, HP_percent, action_established_den=False, died_in_combat=False,
    action=None, scent_dx=0.0, scent_dy=0.0, species_id=None, species_db=None,
    tracking_blood=False, action_raided_den=False
):
    """
    Physiological-aware reward shaping for Overworld Macro DQN:
    - Time Penalty: -0.01 per tick
    - Starvation: -1.0 per tick if biomass == 0
    - Eat/Forage: Reward scales inversely with current fullness (10.0 * (100 - biomass)/100)
    - Reproduce: +1000.0 on ESTABLISH_DEN
    - Lethal Mistake: -100.0 on combat death
    """
    reward = -0.01  # Time penalty
    
    if current_biomass == 0.0:
        reward -= 1.0  # Starvation penalty
        
    forage_bonus = 10.0 * ((100.0 - current_biomass) / 100.0)
    
    if action_established_den:
        reward += 1000.0
        
    if action_raided_den:
        reward += 50.0
        
    if died_in_combat:
        reward -= 100.0
        
    # Dense Tracking Reward for Carnivores/Scavengers
    if species_db and species_id is not None and action is not None:
        spec_info = species_db.get(str(species_id), {})
        diet = spec_info.get("diet")
        if diet in ("Carnivore", "Scavenger") and current_biomass < 80.0:
            if action in (0, 1, 2, 3):  # MOVE_N, MOVE_S, MOVE_E, MOVE_W
                move_dx, move_dy = 0, 0
                if action == 0: move_dy = -1
                elif action == 1: move_dy = 1
                elif action == 2: move_dx = 1
                elif action == 3: move_dx = -1

                if scent_dx != 0.0 or scent_dy != 0.0:
                    orig_dist = math.sqrt(scent_dx**2 + scent_dy**2)
                    new_dist = math.sqrt((scent_dx - move_dx)**2 + (scent_dy - move_dy)**2)
                    
                    if new_dist < orig_dist:
                        track_r = 0.05
                        if tracking_blood:
                            track_r *= 3.0
                        reward += track_r
                    elif new_dist > orig_dist:
                        track_r = 0.05
                        if tracking_blood:
                            track_r *= 3.0
                        reward -= track_r

    return reward, forage_bonus

def get_micro_reward(damage_dealt, target_max_hp, damage_taken, my_max_hp, killed_target=False, escaped=False, hp_percent=1.0):
    """
    Physiological-aware reward shaping for Combat Micro DQN:
    - Turn Penalty: -0.1 per turn
    - Deal Damage: +10.0 * (Damage_Dealt / Target_Max_HP)
    - Take Damage: -10.0 * (Damage_Taken / My_Max_HP)
    - Kill Target: +50.0
    - Flee/Escape (Survival Curve): +50.0 * (1.0 - current_hp_percent)
    """
    reward = -0.1  # Turn penalty
    
    if target_max_hp > 0 and damage_dealt > 0:
        reward += 10.0 * (float(damage_dealt) / float(target_max_hp))
        
    if my_max_hp > 0 and damage_taken > 0:
        reward -= 10.0 * (float(damage_taken) / float(my_max_hp))
        
    if killed_target:
        reward += 50.0
        
    if escaped:
        reward += 15.0 + (35.0 * (1.0 - hp_percent))
        
    return reward
