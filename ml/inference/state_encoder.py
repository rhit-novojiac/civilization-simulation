import torch
from schema.terrain import TileType

def encode_macro_state(monster_id, active_monsters, padded_grid, occupancy_map):
    """
    Converts entity data and local vision grid into a 150-element tensor.
    Input format:
    - 7x7 grid * 3 features (terrain_id, species_id, power_ratio) = 147 values
    - internal stats (hp_percent, biomass, level) = 3 values
    Total = 150 elements.
    """
    monster = active_monsters[monster_id]
    my_species_id, mx, my, hp_percent, level, current_xp, biomass, age, _, _, _, scent_dx, scent_dy, *rest = monster
    
    if occupancy_map is None:
        occupancy_map = {}
        for mid, m in active_monsters.items():
            if mid != monster_id:
                pos = (int(m[1]), int(m[2]))
                if pos not in occupancy_map:
                    occupancy_map[pos] = []
                occupancy_map[pos].append(m)
                
    features = []
    
    # 7x7 Vision Grid centered at (mx, my)
    # my, mx are the true coordinates. In padded_grid (padded by 3), the center is my+3, mx+3.
    # So the slice is my:my+7, mx:mx+7
    int_mx, int_my = int(mx), int(my)
    terrain_slice = padded_grid[int_my:int_my+7, int_mx:int_mx+7]
    
    for dy in range(7):
        for dx in range(7):
            terrain_id = terrain_slice[dy, dx]
            
            # Map back to world coordinates for occupancy map
            tx = int_mx + dx - 3
            ty = int_my + dy - 3
            
            # 2. Species ID & 3. Power Ratio
            occupants = occupancy_map.get((tx, ty))
            if occupants:
                # If multiple, sort by level descending so NN sees most dangerous
                occupants.sort(key=lambda x: x[4], reverse=True)
                other_monster = occupants[0]
                other_species_id = other_monster[0]
                other_level = other_monster[4]
                power_ratio = float(other_level) / float(level)
            else:
                other_species_id = 0
                power_ratio = 0.0
                
            features.extend([float(terrain_id), float(other_species_id), power_ratio])
            
    # Add internal state
    features.append(float(hp_percent))
    features.append(float(biomass) / 100.0) # Normalized biomass (0 to 1)
    
    from config import ConfigManager
    max_level = float(getattr(ConfigManager(), "max_level_cap", 10.0))
    features.append(float(level) / max_level)    # Normalized level (for scaling)
    
    features.append(float(scent_dx))
    features.append(float(scent_dy))
    
    return torch.tensor(features, dtype=torch.float32)

def encode_micro_state(my_monster, target_monster, flee_penalty=0, species_db=None):
    """
    Converts tactical combat state into a 12-element tensor.
    [My_HP, My_STR, My_END, My_DEX, My_AGI, E_HP, E_STR, E_END, E_DEX, E_AGI, Flee_Penalty, Encounter_Tier]
    """
    from engine.combat.physics import get_active_stats, get_max_hp
    import math

    my_stats = get_active_stats(my_monster, species_db)
    t_stats = get_active_stats(target_monster, species_db)
    
    my_max_hp = get_max_hp(my_stats["end"])
    t_max_hp = get_max_hp(t_stats["end"])
    
    my_hp = my_max_hp * my_monster[3] # hp_percent
    t_hp = t_max_hp * target_monster[3]
    
    raw_stats = [
        my_hp, my_stats["str"], my_stats["end"], my_stats["dex"], my_stats["agi"],
        t_hp, t_stats["str"], t_stats["end"], t_stats["dex"], t_stats["agi"]
    ]
    
    max_raw_stat = max(1.0, float(max(raw_stats)))
    
    # Scale all by max_raw_stat
    scaled_stats = [float(val) / max_raw_stat for val in raw_stats]
    
    # Tier Injection
    encounter_tier = min(1.0, math.log10(max_raw_stat) / 3.0)
    
    features = scaled_stats + [float(flee_penalty), encounter_tier]
    
    return torch.tensor(features, dtype=torch.float32)
