import os
import json
import random
from typing import Dict, Tuple

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "config.json"
        )
        self.load_config()
        self._initialized = True

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, "r") as f:
            data = json.load(f)

        # 1. World Config
        world = data.get("world", {})
        self.width = world.get("width", 1000)
        self.height = world.get("height", 1000)
        
        seed = world.get("seed")
        # If seed is set to null or 0 in the config, generate a random seed and write it back
        if seed is None or seed == 0:
            seed = random.randint(1, 10000000)
            print(f"[ConfigManager] Auto-generated random seed: {seed}")
            world["seed"] = seed
            data["world"] = world
            # Save back to config.json
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[ConfigManager] Saved generated seed back to config.json")
            
        self.seed = seed

        # 2. Generation Config
        gen = data.get("generation", {})
        self.island_noise_scale = float(gen.get("island_noise_scale", 100.0))
        self.elevation_noise_scale = float(gen.get("elevation_noise_scale", 50.0))
        self.river_spawn_count = int(gen.get("river_spawn_count", 10))

        # 3. Biomes Config
        biomes = data.get("biomes", {})
        self.seed_counts = biomes.get("seed_counts", {
            "PLAINS": 5,
            "FOREST": 4,
            "JUNGLE": 2,
            "MOUNTAIN": 2
        })
        self.dens_per_biome = int(biomes.get("dens_per_biome", 50))
        
        modifiers = biomes.get("growth_modifiers", {})
        self.forest_water_bonus = float(modifiers.get("forest_water_bonus", 2.0))
        self.jungle_max_elevation = float(modifiers.get("jungle_max_elevation", 0.6))
        self.mountain_min_elevation = float(modifiers.get("mountain_min_elevation", 0.7))
        self.desert_water_penalty = float(modifiers.get("desert_water_penalty", 0.25))
        self.desert_min_water_distance = int(modifiers.get("desert_min_water_distance", 15))
        self.desert_flatland_bonus = float(modifiers.get("desert_flatland_bonus", 2.0))
        self.desert_jungle_buffer = int(modifiers.get("desert_jungle_buffer", 2))

        # Visual Settings
        self.pixels_per_cell = 1 if max(self.width, self.height) >= 1000 else 2
        
        # 4. Simulation Settings
        sim = data.get("simulation", {})
        self.tick_delay_seconds = float(sim.get("tick_delay_seconds", 0.2))
        self.max_population = int(sim.get("max_population", 1000))
        self.max_grazers_per_tile = int(sim.get("max_grazers_per_tile", 10))
        self.initial_monsters_per_den = int(sim.get("initial_monsters_per_den", 1))
        self.den_charges = int(sim.get("den_charges", 10))
        self.epsilon_decay_ticks = int(sim.get("epsilon_decay_ticks", 5000))
        self.max_level_cap = int(sim.get("max_level_cap", 10))
        
        # 5. Logging Config
        logging = data.get("logging", {})
        self.log_ecology = bool(logging.get("log_ecology", True))
        self.log_combat = bool(logging.get("log_combat", True))
        self.log_clock = bool(logging.get("log_clock", True))
        self.log_population = bool(logging.get("log_population", True))
        self.log_world_state = bool(logging.get("log_world_state", True))
        self.log_metabolism = bool(logging.get("log_metabolism", True))
        
        # New Biome Colors mapping to TileType Enum:
        # OCEAN=0, MOUNTAIN=1, FOREST=2, JUNGLE=3, PLAINS=4, FRESHWATER=5, DESERT=6
        self.biome_colors: Dict[int, Tuple[int, int, int]] = {
            0: (15,  94, 156),   # OCEAN:      Deep ocean blue
            1: (160, 160, 168),  # MOUNTAIN:   Cool stone grey
            2: (46,  120,  52),  # FOREST:     Deep forest green
            3: (20,  105,  60),  # JUNGLE:     Dark emerald
            4: (124, 194,  80),  # PLAINS:     Bright meadow green (grassland)
            5: (64,  196, 255),  # FRESHWATER: Clear river cyan
            6: (210, 180, 100)   # DESERT:     Warm sandy gold
        }

    def reload(self):
        """Force a full reload of config.json, resetting the singleton cache."""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        self.load_config()
        self._initialized = True
