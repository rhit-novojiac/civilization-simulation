import uuid
import random
from engine.state_manager import state_manager
from schema.entities import MonsterData, DenData
from schema.terrain import is_walkable, TileType
from engine.logging.metrics import log_life_event

def execute_first_wave(tick, config, species_db):
    """
    Forces every Den in WorldState.dens to instantly fire its spawn logic once,
    populating active_monsters based on config.initial_monsters_per_den.
    """
    dens = state_manager.dens
    if not dens:
        if config.log_population:
            print("[Population] No dens found. First wave skipped.")
        return

    spawned_count = 0
    for den in dens:
        x, y, species_id = den[DenData.X], den[DenData.Y], den[DenData.SPECIES_ID]
        
        for _ in range(config.initial_monsters_per_den):
            # Create a new entity ID
            entity_id = str(uuid.uuid4())
            
            spec_info = species_db.get(str(species_id), {})
            starting_biomass = spec_info.get("starting_biomass", 50.0)
            
            new_monster = [
                species_id,
                x,
                y,
                1.0,  # hp_percent
                1,    # level
                0,    # current_xp
                starting_biomass, # biomass
                0,    # age
                0,    # movement_cooldown
                0,    # has_active_den
                20,   # scent_update_timer (trigger almost immediately)
                0.0,  # scent_dx
                0.0,  # scent_dy
                False, # is_bleeding
                0,     # bleeding_ticks
                False, # tracking_blood
                False  # raided_den
            ]
            
            state_manager.active_monsters[entity_id] = new_monster
            try:
                biome_val = state_manager.get_tile(x, y)
                biome = TileType(biome_val).name
            except Exception:
                biome = "UNKNOWN"
            log_life_event(tick, "birth", species_id, x, y, biome)
            spawned_count += 1
        
    if config.log_population:
        print(f"[Population] First wave complete. Spawned {spawned_count} monsters.")

def spawn_from_dens(tick, config, species_db):
    """
    Called every `config.den_spawn_interval` ticks to spawn monsters from dens.
    For each Den, spawn 1 Level 1 entity of species_id on an adjacent valid tile (checking population limits).
    Subtracts 1 from the Den's charges.
    If charges reach 0, remove the Den.
    """

    dens_to_keep = []
    spawned_count = 0
    
    width = config.width
    height = config.height

    for den in state_manager.dens:
        x, y, species_id = den[DenData.X], den[DenData.Y], den[DenData.SPECIES_ID]
        charges = den[DenData.CHARGES] if len(den) > 3 else 5
        
        creator_id = den[DenData.CREATOR_ID] if len(den) > DenData.CREATOR_ID else None
        
        # Dens persist independently of their creator's status.
        
        # Check population limits before spawning
        if len(state_manager.active_monsters) >= config.max_population:
            dens_to_keep.append([x, y, species_id, charges, creator_id])
            continue
            
        # Find adjacent valid tiles
        valid_adj = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    try:
                        tile = state_manager.get_tile(nx, ny)
                        if is_walkable(TileType(tile)):
                            valid_adj.append((nx, ny))
                    except Exception:
                        pass
        
        if valid_adj:
            # Pick a random valid adjacent tile
            sx, sy = random.choice(valid_adj)
            
            entity_id = str(uuid.uuid4())
            
            spec_info = species_db.get(str(species_id), {})
            starting_biomass = spec_info.get("starting_biomass", 50.0)
            
            new_monster = [
                species_id,
                sx,
                sy,
                1.0,  # hp_percent
                1,    # level
                0,    # current_xp
                starting_biomass, # biomass
                0,    # age
                0,    # movement_cooldown
                0,    # has_active_den
                0,    # scent_update_timer
                0.0,  # scent_dx
                0.0,  # scent_dy
                False, # is_bleeding
                0,     # bleeding_ticks
                False, # tracking_blood
                False  # raided_den
            ]
            state_manager.active_monsters[entity_id] = new_monster
            
            try:
                biome_val = state_manager.get_tile(sx, sy)
                biome = TileType(biome_val).name
            except Exception:
                biome = "UNKNOWN"
            log_life_event(tick, "birth", species_id, sx, sy, biome)
            
            spawned_count += 1
            charges -= 1
            
        if charges > 0:
            dens_to_keep.append([x, y, species_id, charges, creator_id])
        else:
            if creator_id and creator_id in state_manager.active_monsters:
                state_manager.active_monsters[creator_id][MonsterData.HAS_ACTIVE_DEN] = 0
            if config.log_population:
                print(f"[Population] Den at ({x}, {y}) exhausted and removed.")

    state_manager.dens = dens_to_keep
    if spawned_count > 0:
        if config.log_population:
            print(f"[Population] Den spawning cycle complete. Spawned {spawned_count} monsters.")
