import os
import json
from config import ConfigManager

class StateManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(StateManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.base_seed = None
        self.delta_map = {}
        self.altars = []
        self.dens = []
        self.active_monsters = {}
        self.combat_stats = {"kills": 0, "fleds": 0, "draws": 0}
        self.current_tick = 0
        self.base_grid = None  # Cache for generated base grid in RAM

        # Define data paths
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.game_state_path = os.path.join(self.data_dir, "active_game_state.json")
        self.entities_path = os.path.join(self.data_dir, "active_entities.json")

        self._initialized = True

    def initialize_world(self, base_seed: int, world_generator):
        """Called when starting a completely new world."""
        self.base_seed = base_seed
        self.delta_map = {}
        self.altars = []
        self.dens = []
        self.active_monsters = {}
        self.combat_stats = {"kills": 0, "fleds": 0, "draws": 0}
        self.current_tick = 0
        self.rebuild_cache(world_generator)

    def rebuild_cache(self, world_generator):
        if world_generator is not None:
            if ConfigManager().log_world_state:
                print(f"[StateManager] Rebuilding base world cache in RAM using seed {self.base_seed}...")
            self.base_grid = world_generator.generate(self.base_seed)
            if not self.altars:
                self.altars = world_generator.altars.copy()
        else:
            raise ValueError("WorldGenerator instance must be provided to rebuild cache.")

    def get_tile(self, x: int, y: int):
        key = f"{x},{y}"
        if key in self.delta_map:
            return self.delta_map[key]

        if self.base_grid is None:
            raise RuntimeError("Base grid is not built.")

        height, width = self.base_grid.shape
        if not (0 <= x < width and 0 <= y < height):
            raise IndexError(f"Coordinate ({x}, {y}) is out of world bounds ({width}x{height})")

        return int(self.base_grid[y, x])

    def set_tile(self, x: int, y: int, data_obj: dict):
        key = f"{x},{y}"
        self.delta_map[key] = data_obj

    def get_all_entity_states(self):
        return self.active_monsters

    def save_to_disk(self):
        """Saves static game state and volatile entity state to separate files."""
        # 1. Static Game State
        game_state = {
            "base_seed": self.base_seed,
            "delta_map": self.delta_map,
            "altars": self.altars,
            "dens": self.dens,
            "combat_stats": self.combat_stats,
            "current_tick": getattr(self, "current_tick", 0)
        }
        with open(self.game_state_path, "w") as f:
            json.dump(game_state, f, indent=2)
            
        # 2. Volatile Entity State
        with open(self.entities_path, "w") as f:
            json.dump(self.active_monsters, f, indent=2)
            
        if ConfigManager().log_world_state:
            print("[StateManager] Successfully saved game state and entities to disk.")

    def load_from_disk(self, world_generator):
        """Loads both game state and volatile entity state from disk."""
        if os.path.exists(self.game_state_path):
            with open(self.game_state_path, "r") as f:
                game_state = json.load(f)
            self.base_seed = game_state.get("base_seed")
            self.delta_map = game_state.get("delta_map", {})
            self.altars = game_state.get("altars", [])
            self.dens = game_state.get("dens", [])
            self.combat_stats = game_state.get("combat_stats", {"kills": 0, "fleds": 0, "draws": 0})
            self.current_tick = game_state.get("current_tick", 0)
            self.rebuild_cache(world_generator)
            if ConfigManager().log_world_state:
                print("[StateManager] Loaded static game state.")
        else:
            raise FileNotFoundError("active_game_state.json not found.")

        if os.path.exists(self.entities_path):
            with open(self.entities_path, "r") as f:
                self.active_monsters = json.load(f)
                
            # Migration for Age (Schema expanded from 7 to 8 slots)
            for entity_id, entity_data in self.active_monsters.items():
                if len(entity_data) == 7:
                    entity_data.append(0)
                    
            if ConfigManager().log_world_state:
                print("[StateManager] Loaded volatile entity state.")
        else:
            self.active_monsters = {}
            if ConfigManager().log_world_state:
                print("[StateManager] No active entities found (first run).")

# Global singleton instance
state_manager = StateManager()
