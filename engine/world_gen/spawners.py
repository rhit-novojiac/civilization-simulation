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
    
    # Biome Mapping (Weighted Probabilities)
    biome_species_map = {
        TileType.PLAINS: {1: 60, 4: 25, 2: 15},       # Horned Rabbit, Rock Boar, Dire Wolf
        TileType.FOREST: {1: 60, 4: 25, 2: 15},       # Horned Rabbit, Rock Boar, Dire Wolf
        TileType.MOUNTAIN: {4: 80, 2: 20},            # Rock Boar, Dire Wolf
        TileType.JUNGLE: {8: 80, 5: 20},              # Emerald Macaque, Giant Spider
        TileType.DESERT: {7: 80, 6: 20}               # Dune Hopper, Sand Scorpion
    }
    
    height, width = biome_grid.shape
    
    # Radial constraint: restrict to a 275-tile radius from the center
    cy, cx = height / 2.0, width / 2.0
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    radial_mask = (y_grid - cy)**2 + (x_grid - cx)**2 <= 75625
    
    for biome, species_weights in biome_species_map.items():
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
        
        # Filter out deactivated species
        filtered_weights = {k: v for k, v in species_weights.items() if str(k) in config.active_species}
        if not filtered_weights:
            continue
            
        population = list(filtered_weights.keys())
        weights = list(filtered_weights.values())
        
        for i, (y, x) in enumerate(selected_coords):
            species_id = rng.choices(population, weights=weights, k=1)[0]
            dens.append([int(x), int(y), species_id, config.den_charges, None])
            
    if config.log_world_state:
        print(f"[Spawners] Generated {len(dens)} dens across the world.")
    return dens
