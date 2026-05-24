import random
import torch
import math
from engine.state_manager import state_manager
from schema.entities import MonsterData, DenData
from schema.terrain import is_walkable, TileType
from ml.inference.state_encoder import encode_micro_state
from ml.inference.action_decoder import MicroAction
from engine.combat.grid_builder import setup_combat_positions
from engine.combat.physics import (
    resolve_attack,
    apply_xp,
    get_xp_yield,
    get_active_stats,
    get_max_hp
)

def get_micro_action_dx_dy(action):
    """
    Maps MicroAction movement to (dx, dy).
    Returns (0, 0) if BASIC_ATTACK.
    """
    if action == MicroAction.MOVE_N:
        return 0, -1
    elif action == MicroAction.MOVE_S:
        return 0, 1
    elif action == MicroAction.MOVE_E:
        return 1, 0
    elif action == MicroAction.MOVE_W:
        return -1, 0
    elif action == MicroAction.MOVE_NE:
        return 1, -1
    elif action == MicroAction.MOVE_NW:
        return -1, -1
    elif action == MicroAction.MOVE_SE:
        return 1, 1
    elif action == MicroAction.MOVE_SW:
        return -1, 1
    return 0, 0

def resolve_combat(entity_a_id, entity_b_id, species_db, config, brains, epsilon=0.05):
    """
    Resolves combat between entity_a (attacker) and entity_b (defender).
    Runs a turn-based tactical simulation on a 15x15 grid.
    Saves experiences to replay buffers and updates state_manager.active_monsters.
    """
    # Fetch real monsters from state manager
    monsters = state_manager.active_monsters
    if entity_a_id not in monsters or entity_b_id not in monsters:
        return

    entity_a_data = monsters[entity_a_id]
    entity_b_data = monsters[entity_b_id]

    sp_name_a = species_db.get(str(entity_a_data[MonsterData.SPECIES_ID]), {}).get("name", "Unknown")
    sp_name_b = species_db.get(str(entity_b_data[MonsterData.SPECIES_ID]), {}).get("name", "Unknown")

    # Save original overworld coordinates
    ox_a, oy_a = entity_a_data[1], entity_a_data[2]
    ox_b, oy_b = entity_b_data[1], entity_b_data[2]

    # Initialize local combatants on 15x15 grid
    local_a, local_b = setup_combat_positions(entity_a_data, entity_b_data)

    max_turns = 30
    turn = 0
    winner_id = None
    loser_id = None
    fled_id = None

    flee_attempts = {entity_a_id: 0, entity_b_id: 0}

    # Track last states/actions for transition caching
    last_state_a = None
    last_action_a = None
    last_state_b = None
    last_action_b = None

    while turn < max_turns:
        # --- TURN FOR ENTITY A ---
        # Encode tactical state for A targeting B
        state_a = encode_micro_state(local_a, local_b, flee_attempts[entity_a_id], combat_grid_size=15)
        
        # Action selection using Policy Net or random exploration
        spec_id_a = str(local_a[MonsterData.SPECIES_ID])
        model_a = brains[spec_id_a]['micro']
        
        if random.random() < epsilon:
            valid_actions = []
            for act_idx in range(9):
                dx, dy = get_micro_action_dx_dy(act_idx)
                if act_idx == MicroAction.BASIC_ATTACK or (0 <= local_a[1] + dx < 15 and 0 <= local_a[2] + dy < 15) or flee_attempts[entity_a_id] < 3:
                    valid_actions.append(act_idx)
            action_a = random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_vals = model_a(state_a.unsqueeze(0)).clone()
                if flee_attempts[entity_a_id] >= 3:
                    for act_idx in range(8):
                        dx, dy = get_micro_action_dx_dy(act_idx)
                        if not (0 <= local_a[1] + dx < 15 and 0 <= local_a[2] + dy < 15):
                            q_vals[0, act_idx] = -99999.0
                action_a = int(q_vals.argmax(dim=1).item())

        # Resolve A's action
        reward_a = -0.1  # Turn penalty
        damage_dealt_a = 0
        escaped_a = False

        if action_a == MicroAction.BASIC_ATTACK:
            # Check Chebyshev distance
            dist = max(abs(local_b[1] - local_a[1]), abs(local_b[2] - local_a[2]))
            if dist <= 1:
                damage_dealt_a = resolve_attack(local_a, local_b, species_db)
                b_stats = get_active_stats(local_b, species_db)
                b_max_hp = get_max_hp(b_stats["end"])
                
                # --- AMBUSH BONUS ---
                if turn == 0:
                    diet_a = species_db.get(spec_id_a, {}).get("diet", "")
                    diet_b = species_db.get(spec_id_b, {}).get("diet", "")
                    if diet_a == "Carnivore" and diet_b == "Herbivore":
                        ambush_mult = 1.5
                        if spec_id_a == "5": # Giant Spider
                            overworld_terrain = state_manager.get_tile(int(entity_a_data[MonsterData.X]), int(entity_a_data[MonsterData.Y]))
                            if overworld_terrain in (2, 3): # FOREST or JUNGLE
                                ambush_mult = 2.0
                        
                        extra_damage = int(damage_dealt_a * (ambush_mult - 1.0))
                        hp_reduction = float(extra_damage) / float(b_max_hp)
                        local_b[MonsterData.HP_PERCENT] = max(0.0, local_b[MonsterData.HP_PERCENT] - hp_reduction)
                        damage_dealt_a += extra_damage
                # --------------------
                
                reward_a += 10.0 * (float(damage_dealt_a) / float(b_max_hp))
                
                # "Taking a Bite": Carnivores and Scavengers gain +2 biomass per successful hit
                if damage_dealt_a > 0:
                    diet_a = species_db.get(spec_id_a, {}).get("diet", "")
                    if diet_a in ("Carnivore", "Scavenger"):
                        local_a[MonsterData.BIOMASS] = min(100.0, local_a[MonsterData.BIOMASS] + 2.0)
        else:
            dx, dy = get_micro_action_dx_dy(action_a)
            local_a[1] += dx
            local_a[2] += dy
            
            # Check escape with Agility Roll
            if not (0 <= local_a[1] < 15 and 0 <= local_a[2] < 15):
                flee_attempts[entity_a_id] += 1
                a_agi = get_active_stats(local_a, species_db)["agi"]
                b_agi = get_active_stats(local_b, species_db)["agi"]
                if (a_agi + random.randint(1, 20)) >= (b_agi + random.randint(1, 20)):
                    escaped_a = True
                    fled_id = entity_a_id
                else:
                    # Blocked!
                    local_a[1] = max(0, min(14, local_a[1]))
                    local_a[2] = max(0, min(14, local_a[2]))

        # Cache experience for A
        if last_state_a is not None and last_action_a is not None:
            # done is False since turn is not over yet, push previous step transitions
            # To simplify, we will push the transition at the end of the round or turn
            pass

        # Check if B died
        if local_b[MonsterData.HP_PERCENT] <= 0.0:
            winner_id = entity_a_id
            loser_id = entity_b_id
            reward_a += 50.0
            
            xp_yield = get_xp_yield(local_b[MonsterData.LEVEL])
            reward_a += (xp_yield * 0.1)
            
            # Push transition for A (successful kill)
            next_state_a = encode_micro_state(local_a, local_b, flee_attempts[entity_a_id], combat_grid_size=15)
            brains[spec_id_a]['micro_buffer'].push(state_a, action_a, reward_a, next_state_a, True)
            
            # Push terminal punishment for B (death)
            state_b_death = encode_micro_state(local_b, local_a, flee_attempts[entity_b_id], combat_grid_size=15)
            spec_id_b_death = str(local_b[MonsterData.SPECIES_ID])
            brains[spec_id_b_death]['micro_buffer'].push(state_b_death, 0, -100.0, torch.zeros(7), True)
            break

        # Check if A escaped
        if escaped_a:
            reward_a += 50.0 * (1.0 - local_a[MonsterData.HP_PERCENT])
            
            diet_a = species_db.get(spec_id_a, {}).get("diet", "")
            if diet_a == "Herbivore":
                reward_a += (30 * 0.1)
                
            # Push transition for A (escape)
            next_state_a = torch.zeros(7)
            brains[spec_id_a]['micro_buffer'].push(state_a, action_a, reward_a, next_state_a, True)
            break

        # Push standard transition for A
        is_draw_for_A = (turn == max_turns - 1)
        if is_draw_for_A:
            diet_a = species_db.get(spec_id_a, {}).get("diet", "")
            if diet_a == "Herbivore" and flee_attempts[entity_a_id] < 3:
                reward_a += (30 * 0.1)
                
        next_state_a = encode_micro_state(local_a, local_b, flee_attempts[entity_a_id], combat_grid_size=15)
        brains[spec_id_a]['micro_buffer'].push(state_a, action_a, reward_a, next_state_a, is_draw_for_A)

        # --- TURN FOR ENTITY B ---
        state_b = encode_micro_state(local_b, local_a, flee_attempts[entity_b_id], combat_grid_size=15)
        spec_id_b = str(local_b[MonsterData.SPECIES_ID])
        model_b = brains[spec_id_b]['micro']

        if random.random() < epsilon:
            valid_actions = []
            for act_idx in range(9):
                dx, dy = get_micro_action_dx_dy(act_idx)
                if act_idx == MicroAction.BASIC_ATTACK or (0 <= local_b[1] + dx < 15 and 0 <= local_b[2] + dy < 15) or flee_attempts[entity_b_id] < 3:
                    valid_actions.append(act_idx)
            action_b = random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_vals = model_b(state_b.unsqueeze(0)).clone()
                if flee_attempts[entity_b_id] >= 3:
                    for act_idx in range(8):
                        dx, dy = get_micro_action_dx_dy(act_idx)
                        if not (0 <= local_b[1] + dx < 15 and 0 <= local_b[2] + dy < 15):
                            q_vals[0, act_idx] = -99999.0
                action_b = int(q_vals.argmax(dim=1).item())

        reward_b = -0.1  # Turn penalty
        damage_dealt_b = 0
        escaped_b = False

        if action_b == MicroAction.BASIC_ATTACK:
            dist = max(abs(local_a[1] - local_b[1]), abs(local_a[2] - local_b[2]))
            if dist <= 1:
                damage_dealt_b = resolve_attack(local_b, local_a, species_db)
                a_stats = get_active_stats(local_a, species_db)
                a_max_hp = get_max_hp(a_stats["end"])
                reward_b += 10.0 * (float(damage_dealt_b) / float(a_max_hp))
                
                # "Taking a Bite": Carnivores and Scavengers gain +2 biomass per successful hit
                if damage_dealt_b > 0:
                    diet_b = species_db.get(spec_id_b, {}).get("diet", "")
                    if diet_b in ("Carnivore", "Scavenger"):
                        local_b[MonsterData.BIOMASS] = min(100.0, local_b[MonsterData.BIOMASS] + 2.0)
        else:
            dx, dy = get_micro_action_dx_dy(action_b)
            local_b[1] += dx
            local_b[2] += dy
            
            # Check escape with Agility Roll
            if not (0 <= local_b[1] < 15 and 0 <= local_b[2] < 15):
                flee_attempts[entity_b_id] += 1
                b_agi = get_active_stats(local_b, species_db)["agi"]
                a_agi = get_active_stats(local_a, species_db)["agi"]
                if (b_agi + random.randint(1, 20)) >= (a_agi + random.randint(1, 20)):
                    escaped_b = True
                    fled_id = entity_b_id
                else:
                    # Blocked!
                    local_b[1] = max(0, min(14, local_b[1]))
                    local_b[2] = max(0, min(14, local_b[2]))

        # Check if A died
        if local_a[MonsterData.HP_PERCENT] <= 0.0:
            winner_id = entity_b_id
            loser_id = entity_a_id
            reward_b += 50.0
            
            xp_yield = get_xp_yield(local_a[MonsterData.LEVEL])
            reward_b += (xp_yield * 0.1)
            
            next_state_b = encode_micro_state(local_b, local_a, flee_attempts[entity_b_id], combat_grid_size=15)
            brains[spec_id_b]['micro_buffer'].push(state_b, action_b, reward_b, next_state_b, True)
            
            # Push terminal punishment for A (death)
            state_a_death = encode_micro_state(local_a, local_b, flee_attempts[entity_a_id], combat_grid_size=15)
            brains[spec_id_a]['micro_buffer'].push(state_a_death, 0, -100.0, torch.zeros(7), True)
            break

        # Check if B escaped
        if escaped_b:
            reward_b += 50.0 * (1.0 - local_b[MonsterData.HP_PERCENT])
            
            diet_b = species_db.get(spec_id_b, {}).get("diet", "")
            if diet_b == "Herbivore":
                reward_b += (30 * 0.1)
                
            next_state_b = torch.zeros(7)
            brains[spec_id_b]['micro_buffer'].push(state_b, action_b, reward_b, next_state_b, True)
            break

        # Push standard transition for B
        is_draw_for_B = (turn == max_turns - 1)
        if is_draw_for_B:
            diet_b = species_db.get(spec_id_b, {}).get("diet", "")
            if diet_b == "Herbivore" and flee_attempts[entity_b_id] < 3:
                reward_b += (30 * 0.1)
                
        next_state_b = encode_micro_state(local_b, local_a, flee_attempts[entity_b_id], combat_grid_size=15)
        brains[spec_id_b]['micro_buffer'].push(state_b, action_b, reward_b, next_state_b, is_draw_for_B)

        turn += 1

    # --- POST COMBAT RESOLUTION & UPDATES ---
    current_tick = getattr(state_manager, "current_tick", 0)
    
    if winner_id is not None:
        # A death occurred!
        winner_sp_name = sp_name_a if winner_id == entity_a_id else sp_name_b
        loser_sp_name = sp_name_b if winner_id == entity_a_id else sp_name_a
        if config.log_combat:
            print(f"[Combat] {winner_sp_name} ({winner_id[:8]}) defeated {loser_sp_name} ({loser_id[:8]}) in {turn} turns!")
        
        state_manager.combat_stats["kills"] += 1
        
        # 1. Update winner's stats in real active_monsters
        winner_data = monsters[winner_id]
        local_winner = local_a if winner_id == entity_a_id else local_b
        local_loser = local_b if winner_id == entity_a_id else local_a
        
        # Apply XP to winner
        xp_yield = get_xp_yield(local_loser[MonsterData.LEVEL])
        leveled_up = apply_xp(local_winner, xp_yield)
        
        # Dynamic Carnivore biomass boost on kill
        winner_spec_id = str(local_winner[MonsterData.SPECIES_ID])
        winner_spec = species_db.get(winner_spec_id, {})
        if winner_spec.get("diet") == "Carnivore":
            loser_stats = get_active_stats(local_loser, species_db)
            biomass_reward = 10.0 + (loser_stats["end"] * 1.5)
            
            # Anti-Cannibalism Hotfix
            loser_spec_id = str(local_loser[MonsterData.SPECIES_ID])
            if winner_spec_id == loser_spec_id:
                biomass_reward /= 4.0
                
            local_winner[MonsterData.BIOMASS] = min(100.0, local_winner[MonsterData.BIOMASS] + biomass_reward)

        # Copy state back
        winner_data[MonsterData.HP_PERCENT] = local_winner[MonsterData.HP_PERCENT]
        winner_data[MonsterData.LEVEL] = local_winner[MonsterData.LEVEL]
        winner_data[MonsterData.CURRENT_XP] = local_winner[MonsterData.CURRENT_XP]
        winner_data[MonsterData.BIOMASS] = local_winner[MonsterData.BIOMASS]
        
        if config.log_combat:
            loser_sp_name = sp_name_b if winner_id == entity_a_id else sp_name_a
            print(f"[Combat] {winner_sp_name} ({winner_id[:8]}) won the battle, killing {loser_sp_name} ({loser_id[:8]}) in {turn} turns!")
            
        if leveled_up:
            if config.log_combat:
                print(f"[Combat] {winner_sp_name} ({winner_id[:8]}) leveled up to {winner_data[MonsterData.LEVEL]}!")

        # 2. Register carcass for Scavengers (on overworld death tile)
        death_x, death_y = (ox_b, oy_b) if loser_id == entity_b_id else (ox_a, oy_a)
        recent_deaths = getattr(state_manager, "recent_deaths", {})
        recent_deaths[(int(death_x), int(death_y))] = {
            "tick": current_tick,
            "biomass": local_loser[MonsterData.LEVEL] * 20.0
        }
        state_manager.recent_deaths = recent_deaths

        # 3. Delete loser from active monsters
        if loser_id in monsters:
            del monsters[loser_id]
            
    elif fled_id is not None or winner_id is None:
        # Escape or Draw occurred
        if fled_id is not None:
            flee_sp_name = sp_name_a if fled_id == entity_a_id else sp_name_b
            if config.log_combat:
                print(f"[Combat] {flee_sp_name} ({fled_id[:8]}) fled and escaped from the fight in {turn} turns!")
            
            state_manager.combat_stats["fleds"] += 1
            
            # Displace the fleeing entity to an adjacent walkable tile on overworld to prevent infinite loop
            flee_entity = entity_a_data if fled_id == entity_a_id else entity_b_data
            fx, fy = flee_entity[MonsterData.X], flee_entity[MonsterData.Y]
            
            displaced = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nfx = fx + dx
                    nfy = fy + dy
                    if 0 <= nfx < config.width and 0 <= nfy < config.height:
                        try:
                            tile = TileType(state_manager.get_tile(nfx, nfy))
                            if is_walkable(tile):
                                flee_entity[MonsterData.X] = nfx
                                flee_entity[MonsterData.Y] = nfy
                                displaced = True
                                break
                        except Exception:
                            pass
                if displaced:
                    break
        else:
            if config.log_combat:
                print(f"[Combat] Combat between {sp_name_a} ({entity_a_id[:8]}) and {sp_name_b} ({entity_b_id[:8]}) ended in a DRAW.")
            state_manager.combat_stats["draws"] += 1

        # Apply Flee/Draw XP for survivors
        for local_ent, real_ent, sp_id, real_id in [(local_a, entity_a_data, spec_id_a, entity_a_id), (local_b, entity_b_data, spec_id_b, entity_b_id)]:
            diet = species_db.get(sp_id, {}).get("diet", "")
            xp_reward = 30 if (diet == "Herbivore" and (flee_attempts[real_id] < 3 or fled_id == real_id)) else 0
            leveled = apply_xp(local_ent, xp_reward)
            if leveled and config.log_combat:
                name = species_db.get(sp_id, {}).get("name", "Unknown")
                print(f"[Combat] {name} leveled up to {local_ent[MonsterData.LEVEL]} after surviving!")
                
            real_ent[MonsterData.HP_PERCENT] = local_ent[MonsterData.HP_PERCENT]
            real_ent[MonsterData.BIOMASS] = local_ent[MonsterData.BIOMASS]
            real_ent[MonsterData.LEVEL] = local_ent[MonsterData.LEVEL]
            real_ent[MonsterData.CURRENT_XP] = local_ent[MonsterData.CURRENT_XP]
        
        # Displace defender (B) slightly to separate them
        fx, fy = entity_b_data[MonsterData.X], entity_b_data[MonsterData.Y]
        displaced = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nfx = fx + dx
                nfy = fy + dy
                if 0 <= nfx < config.width and 0 <= nfy < config.height:
                    try:
                        tile = TileType(state_manager.get_tile(nfx, nfy))
                        if is_walkable(tile):
                            entity_b_data[MonsterData.X] = nfx
                            entity_b_data[MonsterData.Y] = nfy
                            displaced = True
                            break
                    except Exception:
                        pass
            if displaced:
                break

    outcome = "draw"
    if winner_id is not None:
        outcome = "kill"
    elif fled_id is not None:
        outcome = "flee"
        
    try:
        x = int(entity_a_data[MonsterData.X])
        y = int(entity_a_data[MonsterData.Y])
        try:
            from schema.terrain import TileType
            biome_val = state_manager.get_tile(x, y)
            biome = TileType(biome_val).name
        except Exception:
            biome = "UNKNOWN"
            
        from engine.logging.metrics import log_combat_outcome
        log_combat_outcome(
            current_tick,
            str(entity_a_data[MonsterData.SPECIES_ID]),
            entity_a_data[MonsterData.LEVEL],
            str(entity_b_data[MonsterData.SPECIES_ID]),
            entity_b_data[MonsterData.LEVEL],
            outcome,
            x,
            y,
            biome
        )
    except Exception as e:
        pass

    return winner_id, loser_id

def resolve_overworld_encounters(active_monsters, stances, config, species_db, brains, epsilon):
    """
    Groups entities by overworld position and resolves encounters.
    Returns: (coexistence_rewards, death_flags) dicts mapping entity_ids.
    """
    # Check for Den Poaching
    from engine.state_manager import state_manager
    dens_at_pos = {}
    for idx, den in enumerate(state_manager.dens):
        dens_at_pos[(int(den[0]), int(den[1]))] = idx
        
    for mid, m in list(active_monsters.items()):
        pos = (int(m[1]), int(m[2]))
        stance = stances.get(mid, 0)
        species_id = str(m[MonsterData.SPECIES_ID])
        diet = species_db.get(species_id, {}).get("diet")
        
        if stance == 0 and diet in ("Carnivore", "Scavenger"):
            if pos in dens_at_pos:
                den_idx = dens_at_pos[pos]
                if den_idx is None:
                    continue
                den_species = str(state_manager.dens[den_idx][2])
                den_diet = species_db.get(den_species, {}).get("diet")
                if den_diet == "Herbivore" and species_id != den_species:
                    # Poach!
                    m[MonsterData.BIOMASS] = min(100.0, m[MonsterData.BIOMASS] + 50.0)
                    if config.log_ecology:
                        print(f"[Ecology] {mid[:8]} poached a Den at {pos}!")
                        
                    # Reset creator limit flag
                    poached_den = state_manager.dens[den_idx]
                    creator_id = poached_den[DenData.CREATOR_ID] if len(poached_den) > DenData.CREATOR_ID else None
                    if creator_id and creator_id in active_monsters:
                        active_monsters[creator_id][MonsterData.HAS_ACTIVE_DEN] = 0
                        
                    # Mark Den for removal
                    state_manager.dens[den_idx] = None
                    dens_at_pos[pos] = None # Prevent double poaching
    
    # Remove poached dens
    state_manager.dens = [d for d in state_manager.dens if d is not None]

    grid_occupancy = {}
    for mid, m in list(active_monsters.items()):
        pos = (int(m[1]), int(m[2]))
        if pos not in grid_occupancy:
            grid_occupancy[pos] = []
        grid_occupancy[pos].append(mid)

    coexistence_rewards = {}
    death_flags = {}

    combats_to_run = []
    for pos, entity_ids in grid_occupancy.items():
        if len(entity_ids) >= 2:
            id1, id2 = entity_ids[0], entity_ids[1]
            stance1 = stances.get(id1, 0)
            stance2 = stances.get(id2, 0)
            
            # 1 is PEACEFUL
            if stance1 == 1 and stance2 == 1:
                coexistence_rewards[id1] = 5.0
                coexistence_rewards[id2] = 5.0
                if config.log_ecology:
                    print(f"[Ecology] {id1[:8]} and {id2[:8]} peacefully coexisted!")
            else:
                combats_to_run.append((id1, id2))
                
    for id1, id2 in combats_to_run:
        winner_id, loser_id = resolve_combat(id1, id2, species_db, config, brains, epsilon)
        if loser_id:
            death_flags[loser_id] = True

    return coexistence_rewards, death_flags
