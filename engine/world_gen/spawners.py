import numpy as np
import random
from config import ConfigManager
from schema.terrain import TileType, is_walkable

def populate_dens(config: ConfigManager, biome_grid: np.ndarray, seed: int):
    """
    Day 1 Den Generation (World Initialization).
    Returns a list of dens: [x, y, species_id]
    """
    rng = random.Random(seed + 100)
    dens = []
    
    # Biome Mapping
    biome_species_map = {
        TileType.PLAINS: [1, 3],       # Horned Rabbit, Slime
        TileType.FOREST: [1, 2, 3, 5], # Horned Rabbit, Dire Wolf, Slime, Giant Spider
        TileType.MOUNTAIN: [1, 2, 4],  # Horned Rabbit, Dire Wolf, Rock Boar
        TileType.JUNGLE: [5],          # Giant Spider
        TileType.DESERT: [7, 6]        # Dune Hopper, Sand Scorpion
    }
    
    height, width = biome_grid.shape
    
    # Radial constraint: restrict to a 275-tile radius from the center
    cy, cx = height / 2.0, width / 2.0
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    radial_mask = (y_grid - cy)**2 + (x_grid - cx)**2 <= 75625
    
    for biome, species_list in biome_species_map.items():
        if not is_walkable(biome):
            continue
            
        # Find all coordinates matching this biome and within the radius
        coords = np.argwhere((biome_grid == biome) & radial_mask)
        
        if len(coords) == 0:
            continue
            
        num_to_place = min(config.dens_per_biome, len(coords))
        
        # Use random.sample to instantly pluck valid tiles
        selected_indices = rng.sample(range(len(coords)), num_to_place)
        selected_coords = coords[selected_indices]
        
        # Distribute species evenly among the selected coords for this biome
        for i, (y, x) in enumerate(selected_coords):
            species_id = species_list[i % len(species_list)]
            dens.append([int(x), int(y), species_id, config.den_charges, None])
            
    if config.log_world_state:
        print(f"[Spawners] Generated {len(dens)} dens across the world.")
    return dens
