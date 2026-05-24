def get_macro_reward(current_biomass, HP_percent, action_established_den=False, died_in_combat=False):
    """
    Physiological-aware reward shaping for Overworld Macro DQN:
    - Time Penalty: -0.01 per tick
    - Starvation: -1.0 per tick if biomass == 0
    - Eat/Forage: Reward scales inversely with current fullness (10.0 * (100 - biomass)/100)
    - Reproduce: +50.0 on ESTABLISH_DEN
    - Lethal Mistake: -100.0 on combat death
    """
    reward = -0.01  # Time penalty
    
    if current_biomass == 0.0:
        reward -= 1.0  # Starvation penalty
        
    # Eat/Forage reward (scales inversely with fullness)
    # We only apply this hunger curve scaling if they have recently eaten,
    # but to make it simple we can reward them for having higher biomass or foraging.
    # Actually, the spec says: "Eat/Forage (Hunger Curve): Reward scales inversely with current fullness. A starving monster gets high points; a full monster gets near zero."
    # We can shape this based on whether their biomass increases, or simply as an additive reward when foraging.
    # Let's write it as a helper function that returns the forage reward amount:
    forage_bonus = 10.0 * ((100.0 - current_biomass) / 100.0)
    
    if action_established_den:
        reward += 50.0
        
    if died_in_combat:
        reward -= 100.0
        
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
        reward += 50.0 * (1.0 - hp_percent)
        
    return reward
