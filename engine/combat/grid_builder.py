import random

def setup_combat_positions(entity_a_data, entity_b_data):
    """
    Sets up local combatant states on a 15x15 grid.
    Entity B (defender) spawns at center (7, 7).
    Entity A (attacker) spawns exactly 2 tiles away from the defender,
    simulating an ambush. Direction is based on overworld offset.
    Returns:
        local_a: [species_id, ax, ay, hp_percent, level, current_xp, biomass]
        local_b: [species_id, 7, 7, hp_percent, level, current_xp, biomass]
    """
    # Overworld coordinates
    ax, ay = entity_a_data[1], entity_a_data[2]
    bx, by = entity_b_data[1], entity_b_data[2]
    
    # Calculate offset direction from B to A
    dx = ax - bx
    dy = ay - by
    
    # Clamp offset to [-1, 1] to get unit direction
    cdx = max(-1, min(1, dx)) if dx != 0 else 0
    cdy = max(-1, min(1, dy)) if dy != 0 else 0
    
    # If they are on the exact same tile, pick a random cardinal direction
    if cdx == 0 and cdy == 0:
        direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        cdx, cdy = direction
    
    # Defender at center, attacker 2 tiles away in the offset direction
    local_bx, local_by = 7, 7
    local_ax, local_ay = 7 + (cdx * 2), 7 + (cdy * 2)
    
    # Create local copies of data
    # Format: [species_id, x, y, hp_percent, level, current_xp, biomass]
    local_a = [
        entity_a_data[0],
        local_ax,
        local_ay,
        entity_a_data[3],
        entity_a_data[4],
        entity_a_data[5],
        entity_a_data[6],
        entity_a_data[7],
        entity_a_data[8],
        entity_a_data[9],
        entity_a_data[10],
        entity_a_data[11],
        entity_a_data[12]
    ]
    
    local_b = [
        entity_b_data[0],
        local_bx,
        local_by,
        entity_b_data[3],
        entity_b_data[4],
        entity_b_data[5],
        entity_b_data[6],
        entity_b_data[7],
        entity_b_data[8],
        entity_b_data[9],
        entity_b_data[10],
        entity_b_data[11],
        entity_b_data[12]
    ]
    
    return local_a, local_b
