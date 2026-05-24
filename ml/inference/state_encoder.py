import torch
from schema.terrain import TileType

def encode_macro_state(monster_id, active_monsters, state_manager, width, height, occupancy_map=None):
    """
    Converts entity data and local vision grid into a 150-element tensor.
    Input format:
    - 7x7 grid * 3 features (terrain_id, species_id, power_ratio) = 147 values
    - internal stats (hp_percent, biomass, level) = 3 values
    Total = 150 elements.
    """
    monster = active_monsters[monster_id]
    my_species_id, mx, my, hp_percent, level, current_xp, biomass, age, _, _, _, scent_dx, scent_dy = monster
    
    if occupancy_map is None:
        occupancy_map = {}
        for mid, m in active_monsters.items():
            if mid != monster_id:
                occupancy_map[(int(m[1]), int(m[2]))] = m
                
    features = []
    
    # 7x7 Vision Grid centered at (mx, my)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            tx = int(mx + dx)
            ty = int(my + dy)
            
            # 1. Terrain ID
            if 0 <= tx < width and 0 <= ty < height:
                try:
                    terrain_id = state_manager.get_tile(tx, ty)
                except Exception:
                    terrain_id = 0  # Default to OCEAN if error
            else:
                terrain_id = 0  # OCEAN
                
            # 2. Species ID & 3. Power Ratio
            other_monster = occupancy_map.get((tx, ty))
            if other_monster is not None:
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
    features.append(float(level) / 10.0)    # Normalized level (for scaling)
    features.append(float(scent_dx))
    features.append(float(scent_dy))
    
    return torch.tensor(features, dtype=torch.float32)

def encode_micro_state(my_monster, target_monster, flee_attempts_count=0, combat_grid_size=15):
    """
    Converts tactical combat state into a 7-element tensor.
    - my hp_percent
    - my level (normalized)
    - delta_x (target_x - my_x)
    - delta_y (target_y - my_y)
    - target hp_percent
    - distance_to_nearest_edge (from my position to combat boundary)
    - flee_attempts_remaining (normalized)
    """
    my_species_id, my_x, my_y, my_hp, my_lvl, _, _, _, _, _, _, _, _ = my_monster
    t_species_id, t_x, t_y, t_hp, t_lvl, _, _, _, _, _, _, _, _ = target_monster
    
    dx = float(t_x - my_x)
    dy = float(t_y - my_y)
    
    # Distance to nearest edge of the combat grid
    dist_left = float(my_x)
    dist_right = float(combat_grid_size - 1 - my_x)
    dist_top = float(my_y)
    dist_bottom = float(combat_grid_size - 1 - my_y)
    
    dist_edge = min(dist_left, dist_right, dist_top, dist_bottom)
    
    attempts_remaining_normalized = (3.0 - float(flee_attempts_count)) / 3.0
    
    features = [
        float(my_hp),
        float(my_lvl) / 10.0,
        dx,
        dy,
        float(t_hp),
        dist_edge,
        attempts_remaining_normalized
    ]
    return torch.tensor(features, dtype=torch.float32)
