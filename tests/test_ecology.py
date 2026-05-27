import pytest
from unittest.mock import patch, MagicMock
from schema.entities import MonsterData
from schema.terrain import TileType
from engine.ecology.metabolism import process_metabolism
from engine.state_manager import state_manager

MOCK_SPECIES_DB = {
    "1": {
        "name": "TestHerbivore",
        "diet": "Herbivore",
        "base_lifespan": 10
    },
    "2": {
        "name": "TestScavenger",
        "diet": "Scavenger",
        "base_lifespan": 10
    }
}

class MockConfig:
    log_metabolism = False
    max_grazers_per_tile = 2

def create_mock_monster(species_id, biomass=50.0, hp_percent=1.0, age=0, level=1):
    arr = [0] * 13
    arr[MonsterData.SPECIES_ID] = int(species_id)
    arr[MonsterData.X] = 5
    arr[MonsterData.Y] = 5
    arr[MonsterData.HP_PERCENT] = hp_percent
    arr[MonsterData.LEVEL] = level
    arr[MonsterData.CURRENT_XP] = 0
    arr[MonsterData.BIOMASS] = biomass
    arr[MonsterData.AGE] = age
    return arr

def test_process_metabolism_decay():
    state_manager.active_monsters = {
        "m1": create_mock_monster("1", biomass=10.0)
    }
    state_manager.recent_deaths = {}
    config = MockConfig()
    
    with patch("engine.ecology.metabolism.state_manager.get_tile", return_value=TileType.OCEAN): # 0.1 decay
        process_metabolism(MOCK_SPECIES_DB, config)
        
    assert state_manager.active_monsters["m1"][MonsterData.BIOMASS] == pytest.approx(9.9)

def test_process_metabolism_starvation():
    state_manager.active_monsters = {
        "m1": create_mock_monster("1", biomass=0.0, hp_percent=0.05)
    }
    state_manager.recent_deaths = {}
    config = MockConfig()
    
    with patch("engine.ecology.metabolism.state_manager.get_tile", return_value=TileType.OCEAN):
        process_metabolism(MOCK_SPECIES_DB, config)
        
    # HP goes below 0, should die
    assert "m1" not in state_manager.active_monsters
    assert (5, 5) in state_manager.recent_deaths

def test_process_metabolism_old_age():
    # Base lifespan is 10, level 1 -> max age 10
    state_manager.active_monsters = {
        "m1": create_mock_monster("1", age=10)
    }
    state_manager.recent_deaths = {}
    config = MockConfig()
    
    with patch("engine.ecology.metabolism.state_manager.get_tile", return_value=TileType.OCEAN):
        process_metabolism(MOCK_SPECIES_DB, config)
        
    assert "m1" not in state_manager.active_monsters
    assert (5, 5) in state_manager.recent_deaths

def test_process_metabolism_scavenger():
    state_manager.active_monsters = {
        "m1": create_mock_monster("2", biomass=50.0)
    }
    state_manager.recent_deaths = {
        (5, 5): {"tick": 0, "biomass": 10.0}
    }
    state_manager.current_tick = 1
    config = MockConfig()
    
    with patch("engine.ecology.metabolism.state_manager.get_tile", return_value=TileType.OCEAN):
        process_metabolism(MOCK_SPECIES_DB, config)
        
    # Scavenger eats up to 5.0, minus 0.1 decay = 54.9
    assert state_manager.active_monsters["m1"][MonsterData.BIOMASS] == pytest.approx(54.9)
    assert state_manager.recent_deaths[(5, 5)]["biomass"] == pytest.approx(5.0)

from engine.world_gen.spawners import populate_dens
import numpy as np
from collections import Counter

class MockSpawnerConfig:
    def __init__(self):
        self.dens_per_biome = 200
        self.den_charges = 5
        self.log_world_state = False
        self.log_population = False
        self.active_species = ["1", "2", "3", "4", "5", "6", "7", "8"]

def test_weighted_biome_spawning():
    # Create a 200x200 grid of purely PLAINS (TileType 4)
    biome_grid = np.full((200, 200), TileType.PLAINS.value)
    
    config = MockSpawnerConfig()
    
    # Generate exactly 1000 dens by leveraging the radial mask which spans at least that many tiles
    dens = populate_dens(config, biome_grid, seed=42)
    
    # We should have min(dens_per_biome, num_valid_tiles) dens
    assert len(dens) > 100
    
    species_counts = Counter([den[2] for den in dens])
    
    # We expect roughly 60% Rabbit (1), 25% Rock Boar (4), 15% Dire Wolf (2)
    total = sum(species_counts.values())
    rabbit_ratio = species_counts[1] / total
    boar_ratio = species_counts[4] / total
    wolf_ratio = species_counts[2] / total
    
    # Allow 5% margin of error due to randomness
    assert 0.55 <= rabbit_ratio <= 0.65, f"Rabbit ratio {rabbit_ratio} outside expected bounds"
    assert 0.20 <= boar_ratio <= 0.30, f"Boar ratio {boar_ratio} outside expected bounds"
    assert 0.10 <= wolf_ratio <= 0.20, f"Wolf ratio {wolf_ratio} outside expected bounds"

def test_jungle_spawner_macaques():
    # Create a 200x200 grid of purely JUNGLE (TileType 3)
    biome_grid = np.full((200, 200), TileType.JUNGLE.value)
    config = MockSpawnerConfig()
    
    dens = populate_dens(config, biome_grid, seed=999)
    
    species_counts = Counter([den[2] for den in dens])
    
    # Verify Emerald Macaque (8) exists and is highly populated
    assert 8 in species_counts
    assert species_counts[8] > species_counts[5] # Macaques > Giant Spiders

from engine.ecology.population import spawn_from_dens
from schema.entities import DenData

def test_den_persistence_after_parent_death():
    # 1. Instantiate an entity and a Den
    parent_id = "parent-123"
    state_manager.active_monsters = {
        parent_id: create_mock_monster("1")
    }
    
    # Den format: [x, y, species_id, charges, creator_id]
    den = [5, 5, 1, 5, parent_id]
    state_manager.dens = [den]
    
    config = MockSpawnerConfig()
    config.max_population = 10000
    config.width = 100
    config.height = 100
    
    # 2. Explicitly kill the parent entity
    del state_manager.active_monsters[parent_id]
    assert parent_id not in state_manager.active_monsters
    
    # 3. Trigger spawn cycle
    with patch("engine.ecology.population.state_manager.get_tile", return_value=TileType.PLAINS):
        spawn_from_dens(100, config, MOCK_SPECIES_DB)
        
    # 4. Assert orphaned Den is STILL present and retains charges (minus the one that just spawned)
    assert len(state_manager.dens) == 1
    remaining_den = state_manager.dens[0]
    assert remaining_den[DenData.CREATOR_ID] == parent_id
    assert remaining_den[DenData.CHARGES] == 4

def test_scent_gradient_tracking_reward():
    from ml.training.reward_shaper import get_macro_reward
    
    mock_db = {
        "2": {"name": "CarnivoreTest", "diet": "Carnivore"}
    }
    
    # Target is exactly South of entity (scent_dy = 1.0 from engine's Entity - Target calculation)
    # Actually Target = Entity + (0, 1), so Target - Entity = (0, 1)
    scent_dx = 0.0
    scent_dy = 1.0
    
    # Entity moves South (action 1). Closer! Expect +0.05
    reward_closer, _ = get_macro_reward(
        current_biomass=50.0, HP_percent=1.0, 
        action=1, scent_dx=scent_dx, scent_dy=scent_dy, 
        species_id=2, species_db=mock_db
    )
    
    # Entity moves North (action 0). Farther! Expect -0.05
    reward_farther, _ = get_macro_reward(
        current_biomass=50.0, HP_percent=1.0, 
        action=0, scent_dx=scent_dx, scent_dy=scent_dy, 
        species_id=2, species_db=mock_db
    )
    
    # Entity is FULL (biomass 90.0). Should ignore tracking reward.
    reward_full, _ = get_macro_reward(
        current_biomass=90.0, HP_percent=1.0, 
        action=1, scent_dx=scent_dx, scent_dy=scent_dy, 
        species_id=2, species_db=mock_db
    )
    
    # Expected base reward: -0.01 (Time penalty)
    assert reward_closer == -0.01 + 0.05
    assert reward_farther == -0.01 - 0.05
    assert reward_full == -0.01

def test_active_species_filter():
    from engine.world_gen.spawners import populate_dens
    from schema.terrain import TileType
    from schema.entities import DenData
    import numpy as np
    
    config = MockConfig()
    config.active_species = ["2", "4"] # Only Dire Wolf and Rock Boar allowed
    config.dens_per_biome = 10
    config.den_charges = 3
    config.log_world_state = False
    
    # Create a dummy biome grid with all TileTypes
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[0:2, 0:2] = TileType.PLAINS
    grid[2:4, 0:2] = TileType.FOREST
    grid[4:6, 0:2] = TileType.MOUNTAIN
    grid[6:8, 0:2] = TileType.JUNGLE
    grid[8:10, 0:2] = TileType.DESERT
    
    dens = populate_dens(config, grid, seed=42)
    
    # Assert that no dens were spawned for anything other than species 2 and 4
    for den in dens:
        assert str(den[DenData.SPECIES_ID]) in ["2", "4"]
        
    # Since Jungle and Desert only contain species 5, 6, 7, 8
    # No dens should be placed in those biomes!
    for den in dens:
        x, y = den[DenData.X], den[DenData.Y]
        biome = grid[y, x]
        assert biome not in (TileType.JUNGLE, TileType.DESERT)
