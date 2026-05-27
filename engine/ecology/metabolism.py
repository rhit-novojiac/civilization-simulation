from engine.state_manager import state_manager
from schema.entities import MonsterData
from schema.terrain import TileType
from engine.logging.metrics import log_life_event

def process_metabolism(species_db, config):
    """
    Applies biomass decay and starvation penalties to all active monsters.
    Removes entities that die from starvation.
    Also manages scavenger biomass accumulation.
    """
    recent_deaths = getattr(state_manager, "recent_deaths", {})
    current_tick = getattr(state_manager, "current_tick", 0)
    
    # Filter out deaths older than 20 ticks (Background Rot)
    expired_deaths = [pos for pos, carcass in list(recent_deaths.items()) 
                      if isinstance(carcass, dict) and current_tick - carcass["tick"] > 20
                      or not isinstance(carcass, dict) and current_tick - carcass > 20]
    for pos in expired_deaths:
        del recent_deaths[pos]
        
    # 1. Pass 1: Pre-compute spatial occupancy for overgrazing checks
    occupancy_counts = {}
    for _, entity_data in state_manager.active_monsters.items():
        pos = (int(entity_data[MonsterData.X]), int(entity_data[MonsterData.Y]))
        occupancy_counts[pos] = occupancy_counts.get(pos, 0) + 1

    # 2. Pass 2: Process regular metabolism and starvation (Corpse Generation)
    to_delete = []
    old_age_count = 0
    overgrazing_penalties = {}
    
    for entity_id, entity_data in state_manager.active_monsters.items():
        x = entity_data[MonsterData.X]
        y = entity_data[MonsterData.Y]
        biomass = entity_data[MonsterData.BIOMASS]
        hp_percent = entity_data[MonsterData.HP_PERCENT]
        level = entity_data[MonsterData.LEVEL]
        
        # Increment Age
        entity_data[MonsterData.AGE] += 1
        
        # Check Old Age
        species_id = str(entity_data[MonsterData.SPECIES_ID])
        spec_info = species_db.get(species_id, {})
        base_lifespan = spec_info.get("base_lifespan", 500)
        max_age = base_lifespan * level
        
        if entity_data[MonsterData.AGE] >= max_age:
            to_delete.append(entity_id)
            old_age_count += 1
            try:
                biome_val = state_manager.get_tile(x, y)
                biome = TileType(biome_val).name
            except Exception:
                biome = "UNKNOWN"
            log_life_event(current_tick, "death_old_age", species_id, x, y, biome)
            # Leave a level-based carcass
            recent_deaths[(int(x), int(y))] = {
                "tick": current_tick,
                "biomass": level * 20.0
            }
            continue
        
        # Check current tile for biome modifiers
        tile_type = state_manager.get_tile(x, y)
        
        # --- NEW PASSIVE GRAZING LOGIC ---
        if spec_info.get("diet") == "Herbivore":
            grazing_gain = 0.0
            
            # OVERGRAZING CHECK
            if occupancy_counts.get((int(x), int(y)), 1) <= config.max_grazers_per_tile:
                if tile_type == TileType.PLAINS:
                    grazing_gain = 0.5
                elif species_id == "1" and tile_type in (TileType.FOREST, TileType.MOUNTAIN):
                    grazing_gain = 0.5
                elif species_id == "4" and tile_type == TileType.MOUNTAIN:
                    grazing_gain = 0.5
                elif species_id == "7" and tile_type == TileType.DESERT:
                    grazing_gain = 0.5
                elif species_id == "8" and tile_type in (TileType.FOREST, TileType.JUNGLE):
                    grazing_gain = 0.5
            else:
                overgrazing_penalties[entity_id] = -2.0
                
            biomass = min(100.0, biomass + grazing_gain)
        # ---------------------------------
        
        decay_rate = 0.1
        if tile_type == TileType.DESERT:
            decay_rate = 0.2
            
        new_biomass = max(0.0, biomass - decay_rate)
        entity_data[MonsterData.BIOMASS] = new_biomass
        
        # Starvation Penalty
        if new_biomass == 0.0:
            hp_percent -= 0.10
            if hp_percent <= 0.0:
                to_delete.append(entity_id)
                try:
                    biome_val = state_manager.get_tile(x, y)
                    biome = TileType(biome_val).name
                except Exception:
                    biome = "UNKNOWN"
                log_life_event(current_tick, "death_starvation", species_id, x, y, biome)
                # Register death position and tick
                recent_deaths[(int(x), int(y))] = {
                    "tick": current_tick,
                    "biomass": level * 10.0  # Starved entities drop less biomass
                }
            else:
                entity_data[MonsterData.HP_PERCENT] = hp_percent

    # 3. Pass 3: Process Scavenger feeding (Now correctly accessing fresh corpses)
    scavenger_fed_count = 0
    for entity_id, entity_data in state_manager.active_monsters.items():
        if entity_id in to_delete:
            continue
            
        species_id = str(entity_data[MonsterData.SPECIES_ID])
        spec_info = species_db.get(species_id)
        if spec_info and spec_info.get("diet") == "Scavenger":
            x = int(entity_data[MonsterData.X])
            y = int(entity_data[MonsterData.Y])
            
            if (x, y) in recent_deaths:
                carcass = recent_deaths[(x, y)]
                if isinstance(carcass, dict):
                    # Finite Biomass Pool Logic
                    available = carcass["biomass"]
                    amount_to_eat = min(5.0, available)
                    
                    if amount_to_eat > 0:
                        entity_data[MonsterData.BIOMASS] = min(100.0, entity_data[MonsterData.BIOMASS] + amount_to_eat)
                        carcass["biomass"] -= amount_to_eat
                        scavenger_fed_count += 1
                        
                        if carcass["biomass"] <= 0:
                            del recent_deaths[(x, y)]
                else:
                    # Legacy support if any old int timestamps remain
                    entity_data[MonsterData.BIOMASS] = min(100.0, entity_data[MonsterData.BIOMASS] + 5.0)
                    scavenger_fed_count += 1
                    
    if scavenger_fed_count > 0 and config.log_metabolism:
        print(f"[Metabolism] {scavenger_fed_count} Scavengers fed on carcasses.")
        
    # 4. Safe Sweep: Cleanup dead entities
    for entity_id in to_delete:
        if entity_id in state_manager.active_monsters:
            del state_manager.active_monsters[entity_id]
            
    state_manager.recent_deaths = recent_deaths
        
    if to_delete and config.log_metabolism:
        starved_count = len(to_delete) - old_age_count
        if starved_count > 0:
            print(f"[Metabolism] {starved_count} entities starved to death this tick.")
        if old_age_count > 0:
            print(f"[Metabolism] {old_age_count} entities died of old age.")
            
    return overgrazing_penalties
