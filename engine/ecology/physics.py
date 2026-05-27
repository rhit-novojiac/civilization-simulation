from engine.state_manager import state_manager
from schema.entities import MonsterData, DenData
from schema.terrain import is_walkable, TileType
from ml.inference.action_decoder import MacroAction
import math

TERRAIN_COOLDOWNS = {
    TileType.PLAINS: 0,
    TileType.DESERT: 0,
    TileType.FOREST: 1,
    TileType.MOUNTAIN: 2,
    TileType.JUNGLE: 2
}

def update_scent_compass(state_manager, species_db):
    """
    Calculates normalized directional vectors for Carnivores pointing towards the nearest bleeding Herbivore,
    falling back to the nearest Herbivore Den if no bleeding entities are within 50 tiles.
    """
    # Pre-filter herbivore dens and bleeding herbivores
    herbivore_dens = []
    for den in state_manager.dens:
        den_sp = str(den[DenData.SPECIES_ID])
        if species_db.get(den_sp, {}).get("diet") == "Herbivore":
            herbivore_dens.append((den[DenData.X], den[DenData.Y]))

    bleeding_herbivores = []
    for m in state_manager.active_monsters.values():
        sp = str(m[MonsterData.SPECIES_ID])
        if species_db.get(sp, {}).get("diet") == "Herbivore" and m[MonsterData.IS_BLEEDING]:
            bleeding_herbivores.append((m[MonsterData.X], m[MonsterData.Y]))

    for mid, m in state_manager.active_monsters.items():
        species_id = str(m[MonsterData.SPECIES_ID])
        diet = species_db.get(species_id, {}).get("diet")
        
        if diet == "Carnivore":
            if m[MonsterData.SCENT_UPDATE_TIMER] > 0:
                m[MonsterData.SCENT_UPDATE_TIMER] -= 1
            else:
                best_sq_dist = float('inf')
                best_dx, best_dy = 0.0, 0.0
                my_x, my_y = m[MonsterData.X], m[MonsterData.Y]
                
                # 1. Check for bleeding prey (radius 50 tiles, squared = 2500)
                found_blood = False
                for bx, by in bleeding_herbivores:
                    dx = bx - my_x
                    dy = by - my_y
                    sq_dist = dx*dx + dy*dy
                    if sq_dist <= 2500.0 and sq_dist < best_sq_dist:
                        best_sq_dist = sq_dist
                        best_dx, best_dy = dx, dy
                        found_blood = True
                
                # 2. Fallback to Dens if no blood trail
                if not found_blood:
                    if not herbivore_dens:
                        m[MonsterData.SCENT_DX] = 0.0
                        m[MonsterData.SCENT_DY] = 0.0
                        m[MonsterData.SCENT_UPDATE_TIMER] = 20
                        m[MonsterData.TRACKING_BLOOD] = False
                        continue
                        
                    for den_x, den_y in herbivore_dens:
                        dx = den_x - my_x
                        dy = den_y - my_y
                        sq_dist = dx*dx + dy*dy
                        if sq_dist < best_sq_dist:
                            best_sq_dist = sq_dist
                            best_dx, best_dy = dx, dy

                m[MonsterData.TRACKING_BLOOD] = found_blood
                
                if best_sq_dist > 0:
                    dist = math.sqrt(best_sq_dist)
                    m[MonsterData.SCENT_DX] = float(best_dx) / dist
                    m[MonsterData.SCENT_DY] = float(best_dy) / dist
                else:
                    m[MonsterData.SCENT_DX] = 0.0
                    m[MonsterData.SCENT_DY] = 0.0
                    
                m[MonsterData.SCENT_UPDATE_TIMER] = 20

def apply_macro_action(entity_id, action, config, species_db):
    """
    Applies a MacroAction to an active monster on the overworld.
    Handles boundaries and walkability collisions.
    """
    entity_data = state_manager.active_monsters.get(entity_id)
    if not entity_data:
        return

    species_id = entity_data[MonsterData.SPECIES_ID]
    x = entity_data[MonsterData.X]
    y = entity_data[MonsterData.Y]
    hp_percent = entity_data[MonsterData.HP_PERCENT]
    level = entity_data[MonsterData.LEVEL]
    current_xp = entity_data[MonsterData.CURRENT_XP]
    biomass = entity_data[MonsterData.BIOMASS]

    width = config.width
    height = config.height

    dx, dy = 0, 0
    if action == MacroAction.MOVE_N:
        dy = -1
    elif action == MacroAction.MOVE_S:
        dy = 1
    elif action == MacroAction.MOVE_E:
        dx = 1
    elif action == MacroAction.MOVE_W:
        dx = -1
    elif action == MacroAction.REST:
        # HP replenishment
        entity_data[MonsterData.HP_PERCENT] = min(1.0, hp_percent + 0.1)
        return
    elif action == MacroAction.ESTABLISH_DEN:
        if entity_data[MonsterData.HAS_ACTIVE_DEN]:
            return
            
        # reproduction action
        spec_info = species_db.get(str(species_id), {})
        species_reproduction_threshold = spec_info.get("reproduction_threshold", 99.0)
        
        if biomass >= species_reproduction_threshold:
            starting_biomass = spec_info.get("starting_biomass", 50.0)
            entity_data[MonsterData.BIOMASS] = starting_biomass
            entity_data[MonsterData.HAS_ACTIVE_DEN] = 1
            state_manager.dens.append([int(x), int(y), int(species_id), config.den_charges, entity_id])
            if config.log_ecology:
                print(f"[Ecology] Monster {entity_id[:8]} (Species {species_id}) established a Den at ({x}, {y})")
        return

    nx = x + dx
    ny = y + dy

    # Collision & Boundary check
    if 0 <= nx < width and 0 <= ny < height:
        try:
            tile = TileType(state_manager.get_tile(nx, ny))
            if is_walkable(tile):
                entity_data[MonsterData.X] = nx
                entity_data[MonsterData.Y] = ny
                
                # Apply Terrain Cooldowns
                cooldown = TERRAIN_COOLDOWNS.get(tile, 0)
                # Species Affinities Overrides
                if str(species_id) == "5" and tile in (TileType.FOREST, TileType.JUNGLE):
                    cooldown = 0
                elif str(species_id) in ("2", "4") and tile == TileType.MOUNTAIN:
                    cooldown = 0
                elif str(species_id) in ("6", "7") and tile == TileType.DESERT:
                    cooldown = 0
                
                entity_data[MonsterData.MOVEMENT_COOLDOWN] = cooldown
                pass
        except Exception:
            pass  # Remain in place if error
