import random
import torch
import math
from engine.state_manager import state_manager
from schema.entities import MonsterData, DenData
from schema.terrain import is_walkable, TileType
from ml.inference.state_encoder import encode_micro_state
from ml.inference.action_decoder import MicroAction
from engine.combat.physics import (
    resolve_attack,
    apply_xp,
    get_xp_yield,
    get_active_stats,
    get_max_hp
)

def resolve_combat(entity_list, species_db, config, brains, epsilon=0.05, pretrained_species=None):
    """
    Abstract Auto-Battler N-vs-N resolution loop.
    Returns a list of dead entity IDs.
    """
    monsters = state_manager.active_monsters
    combatants = [eid for eid in entity_list if eid in monsters]
    if len(combatants) < 2:
        return []
        
    if pretrained_species is None:
        pretrained_species = set()
        
    # Action Economy Penalty
    for eid in combatants:
        monsters[eid][MonsterData.MOVEMENT_COOLDOWN] = max(1, monsters[eid][MonsterData.MOVEMENT_COOLDOWN])
        
    initiator = combatants[-1] # The entity whose movement caused the collision
    
    # [OPTIMIZATION] Pre-compute Agility since level-ups and stat changes only happen post-combat
    agi_cache = {eid: get_active_stats(monsters[eid], species_db)["agi"] for eid in combatants}
        
    sorted_queue = sorted(combatants, key=lambda e: agi_cache[e], reverse=True)
    
    dead_ids = []
    fled_ids = []
    flee_penalties = {eid: 0 for eid in combatants}
    flat_footed = {eid: False for eid in combatants}
    initial_hp = {eid: monsters[eid][MonsterData.HP_PERCENT] for eid in combatants}
    
    last_states = {eid: None for eid in combatants}
    last_actions = {eid: None for eid in combatants}
    last_targets = {eid: None for eid in combatants}
    
    round_num = 0
    max_rounds = 100
    
    def get_action(eid):
        m = monsters[eid]
        spec_id = str(m[MonsterData.SPECIES_ID])
        model = brains[spec_id]['micro']
        
        alive_enemies = [
            e for e in combatants 
            if e != eid 
            and e not in dead_ids 
            and e not in fled_ids
            and str(monsters[e][MonsterData.SPECIES_ID]) != spec_id
        ]
        if not alive_enemies:
            return MicroAction.TOLERATE, None, torch.zeros(12)
            
        alive_enemies.sort(key=lambda e: monsters[e][MonsterData.HP_PERCENT])
        target_id = alive_enemies[0]
        state_tensor = encode_micro_state(m, monsters[target_id], flee_penalties[eid], species_db)
        
        active_epsilon = 0.05 if spec_id in pretrained_species else epsilon
        
        if random.random() < active_epsilon:
            action = random.choice([MicroAction.BASIC_ATTACK, MicroAction.FLEE, MicroAction.TOLERATE])
        else:
            with torch.no_grad():
                q_vals = model(state_tensor.unsqueeze(0))
                action = int(q_vals.argmax(dim=1).item())
                
        return action, target_id, state_tensor

    def push_transition(eid, reward, next_state, done):
        if last_states[eid] is not None and last_actions[eid] is not None:
            spec_id = str(monsters[eid][MonsterData.SPECIES_ID])
            brains[spec_id]['micro_buffer'].push(last_states[eid], last_actions[eid], reward, next_state, done)

    # Turn 0 (Initiator Advantage)
    if initiator in combatants:
        action, target_id, state_tensor = get_action(initiator)
        last_states[initiator] = state_tensor
        last_actions[initiator] = action
        last_targets[initiator] = target_id
        
        if action == MicroAction.BASIC_ATTACK and target_id:
            t_x, t_y = int(monsters[initiator][MonsterData.X]), int(monsters[initiator][MonsterData.Y])
            terrain_val = state_manager.get_tile(t_x, t_y)
            hit, dmg = resolve_attack(
                monsters[initiator], 
                monsters[target_id], 
                species_db, 
                is_flat_footed=flat_footed[target_id],
                is_turn_zero=True,
                terrain_val=terrain_val
            )
            # Gain biomass if Carnivore
            diet = species_db.get(str(monsters[initiator][MonsterData.SPECIES_ID]), {}).get("diet")
            if hit and diet in ("Carnivore", "Scavenger"):
                monsters[initiator][MonsterData.BIOMASS] = min(100.0, monsters[initiator][MonsterData.BIOMASS] + 2.0)
            
            if monsters[target_id][MonsterData.HP_PERCENT] <= 0.0:
                dead_ids.append(target_id)
        elif action == MicroAction.FLEE:
            my_agi = agi_cache[initiator]
            enemy_agi = max([agi_cache[e] for e in combatants if e != initiator and e not in dead_ids and e not in fled_ids] or [0])
            if (my_agi + random.randint(1, 20) + flee_penalties[initiator]) >= (enemy_agi + random.randint(1, 20)):
                fled_ids.append(initiator)
            else:
                flee_penalties[initiator] -= 2
        elif action == MicroAction.TOLERATE:
            flat_footed[initiator] = True

    # Main Resolution Loop
    while round_num < max_rounds:
        round_actions = {}
        
        for eid in sorted_queue:
            if eid in dead_ids or eid in fled_ids:
                continue
                
            action, target_id, state_tensor = get_action(eid)
            round_actions[eid] = action
            flat_footed[eid] = False # Reset before taking action
            
            # Push intermediate transition for previous round
            if last_states[eid] is not None:
                reward = -0.1 # Turn penalty
                push_transition(eid, reward, state_tensor, False)
                
            last_states[eid] = state_tensor
            last_actions[eid] = action
            last_targets[eid] = target_id
            
            if action == MicroAction.BASIC_ATTACK and target_id:
                hit, dmg = resolve_attack(monsters[eid], monsters[target_id], species_db, is_flat_footed=flat_footed[target_id])
                diet = species_db.get(str(monsters[eid][MonsterData.SPECIES_ID]), {}).get("diet")
                if hit and diet in ("Carnivore", "Scavenger"):
                    monsters[eid][MonsterData.BIOMASS] = min(100.0, monsters[eid][MonsterData.BIOMASS] + 2.0)
                
                if monsters[target_id][MonsterData.HP_PERCENT] <= 0.0:
                    dead_ids.append(target_id)
            elif action == MicroAction.FLEE:
                my_agi = agi_cache[eid]
                enemy_agi = max([agi_cache[e] for e in combatants if e != eid and e not in dead_ids and e not in fled_ids] or [0])
                if (my_agi + random.randint(1, 20) + flee_penalties[eid]) >= (enemy_agi + random.randint(1, 20)):
                    fled_ids.append(eid)
                else:
                    flee_penalties[eid] -= 2
            elif action == MicroAction.TOLERATE:
                flat_footed[eid] = True
                
        # Check for Staredown
        alive_combatants = [e for e in combatants if e not in dead_ids and e not in fled_ids]
        if len(alive_combatants) <= 1:
            break
            
        all_tolerate = True
        for e in alive_combatants:
            if round_actions.get(e) != MicroAction.TOLERATE:
                all_tolerate = False
                break
                
        if all_tolerate:
            break
            
        round_num += 1

    # Fetch combat context for logging
    from engine.logging.metrics import log_combat_outcome
    from schema.terrain import TileType
    current_tick = getattr(state_manager, "current_tick", 0)
    
    try:
        init_m = monsters.get(initiator)
        if init_m:
            log_x, log_y = int(init_m[MonsterData.X]), int(init_m[MonsterData.Y])
            biome_val = state_manager.get_tile(log_x, log_y)
            biome = TileType(biome_val).name
        else:
            log_x, log_y, biome = 0, 0, "UNKNOWN"
    except Exception:
        log_x, log_y, biome = 0, 0, "UNKNOWN"

    # Apply terminal rewards and push terminal transitions
    for eid in combatants:
        if last_states[eid] is None:
            continue
            
        m = monsters.get(eid)
        if not m:
            continue
            
        if eid in dead_ids:
            push_transition(eid, -100.0, torch.zeros(12), True)
            
            # Grant XP and biomass to the killer (who targeted this entity)
            for killer_id in combatants:
                if killer_id not in dead_ids and last_targets[killer_id] == eid and last_actions[killer_id] == MicroAction.BASIC_ATTACK:
                    xp_yield = get_xp_yield(m[MonsterData.LEVEL])
                    apply_xp(monsters[killer_id], xp_yield)
                    
                    killer_spec = species_db.get(str(monsters[killer_id][MonsterData.SPECIES_ID]), {})
                    killer_diet = killer_spec.get("diet", "Unknown")
                    if killer_diet == "Carnivore":
                        loser_stats = get_active_stats(m, species_db)
                        biomass_reward = math.floor(10.0 + (loser_stats["end"] * 1.5))
                        monsters[killer_id][MonsterData.BIOMASS] = min(100.0, monsters[killer_id][MonsterData.BIOMASS] + biomass_reward)
                        push_transition(killer_id, 100.0, torch.zeros(12), True)
                    else:
                        push_transition(killer_id, 50.0, torch.zeros(12), True)
                    
                    if getattr(config, "log_combat", False):
                        log_combat_outcome(
                            current_tick, 
                            str(monsters[killer_id][MonsterData.SPECIES_ID]), 
                            monsters[killer_id][MonsterData.LEVEL],
                            str(m[MonsterData.SPECIES_ID]), 
                            m[MonsterData.LEVEL], 
                            "kill", log_x, log_y, biome
                        )
                    break
        elif eid in fled_ids:
            my_spec = species_db.get(str(m[MonsterData.SPECIES_ID]), {})
            if my_spec.get("diet") == "Herbivore":
                reward = 15.0 + (35.0 * (1.0 - m[MonsterData.HP_PERCENT]))
                if m[MonsterData.HP_PERCENT] < initial_hp[eid]:
                    m[MonsterData.IS_BLEEDING] = True
                    m[MonsterData.BLEEDING_TICKS] = 50
            else:
                reward = 0.0
            push_transition(eid, reward, torch.zeros(12), True)
            
            if getattr(config, "log_combat", False):
                b_id = last_targets.get(eid)
                if b_id and b_id in monsters:
                    b_m = monsters[b_id]
                    log_combat_outcome(
                        current_tick, 
                        str(m[MonsterData.SPECIES_ID]), 
                        m[MonsterData.LEVEL],
                        str(b_m[MonsterData.SPECIES_ID]), 
                        b_m[MonsterData.LEVEL], 
                        "flee", log_x, log_y, biome
                    )
            
            state_manager.combat_stats["fleds"] += 1
            
            # Displace entity
            fx, fy = m[MonsterData.X], m[MonsterData.Y]
            displaced = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nfx, nfy = fx + dx, fy + dy
                    if 0 <= nfx < getattr(config, "width", 100) and 0 <= nfy < getattr(config, "height", 100):
                        try:
                            tile = TileType(state_manager.get_tile(nfx, nfy))
                            if is_walkable(tile):
                                m[MonsterData.X], m[MonsterData.Y] = nfx, nfy
                                displaced = True
                                break
                        except Exception:
                            pass
                if displaced:
                    break
        else:
            # Draw / Survive
            my_spec = species_db.get(str(m[MonsterData.SPECIES_ID]), {})
            if my_spec.get("diet") == "Herbivore":
                reward = 50.0
            elif my_spec.get("diet") == "Carnivore":
                reward = 0.0
            else:
                reward = 5.0
            push_transition(eid, reward, torch.zeros(12), True)
            state_manager.combat_stats["draws"] += 1
            if getattr(config, "log_combat", False):
                b_id = last_targets.get(eid)
                if b_id and b_id in monsters:
                    b_m = monsters[b_id]
                    log_combat_outcome(
                        current_tick, 
                        str(m[MonsterData.SPECIES_ID]), 
                        m[MonsterData.LEVEL],
                        str(b_m[MonsterData.SPECIES_ID]), 
                        b_m[MonsterData.LEVEL], 
                        "draw", log_x, log_y, biome
                    )

    # Clean up dead
    for dead_id in dead_ids:
        if dead_id in monsters:
            m = monsters[dead_id]
            recent_deaths = getattr(state_manager, "recent_deaths", {})
            recent_deaths[(int(m[MonsterData.X]), int(m[MonsterData.Y]))] = {
                "tick": getattr(state_manager, "current_tick", 0),
                "biomass": m[MonsterData.LEVEL] * 20.0
            }
            state_manager.recent_deaths = recent_deaths
            del monsters[dead_id]
            state_manager.combat_stats["kills"] += 1
            
    return dead_ids

def resolve_overworld_encounters(active_monsters, stances, config, species_db, brains, epsilon, pretrained_species=None):
    """
    Groups entities by overworld position and resolves encounters using Intent Filter.
    Returns: (coexistence_rewards, death_flags) dicts mapping entity_ids.
    """
    dens_at_pos = {}
    for idx, den in enumerate(state_manager.dens):
        dens_at_pos[(int(den[0]), int(den[1]))] = idx
        
    grid_occupancy = {}
    hostile_tiles = set()
    diet_cache = {}
    for sp_id, spec_info in species_db.items():
        diet_cache[str(sp_id)] = spec_info.get("diet")
        
    for mid, m in active_monsters.items():
        pos = (int(m[1]), int(m[2]))
        
        if pos not in grid_occupancy:
            grid_occupancy[pos] = []
        grid_occupancy[pos].append(mid)
        
        stance = stances.get(mid, 0)
        species_id = str(m[MonsterData.SPECIES_ID])
        diet = diet_cache.get(species_id)
        
        # [OPTIMIZATION] Single-pass O(1) Intent Filter
        if diet in ("Carnivore", "Scavenger") and m[MonsterData.BIOMASS] < 70.0:
            hostile_tiles.add(pos)
        
        if stance == 0 and diet in ("Carnivore", "Scavenger"):
            if pos in dens_at_pos:
                den_idx = dens_at_pos[pos]
                if den_idx is None:
                    continue
                den_species = str(state_manager.dens[den_idx][2])
                den_diet = diet_cache.get(den_species)
                if den_diet == "Herbivore" and species_id != den_species:
                    m[MonsterData.BIOMASS] = min(100.0, m[MonsterData.BIOMASS] + 50.0)
                    poached_den = state_manager.dens[den_idx]
                    creator_id = poached_den[DenData.CREATOR_ID] if len(poached_den) > DenData.CREATOR_ID else None
                    if creator_id and creator_id in active_monsters:
                        active_monsters[creator_id][MonsterData.HAS_ACTIVE_DEN] = 0
                    state_manager.dens[den_idx] = None
                    dens_at_pos[pos] = None
    
    state_manager.dens = [d for d in state_manager.dens if d is not None]

    coexistence_rewards = {}
    death_flags = {}
    combats_to_run = []
    
    for pos, entity_ids in grid_occupancy.items():
        if len(entity_ids) >= 2:
            if pos not in hostile_tiles:
                # Peaceful Coexistence (Reward Removed)
                pass
            else:
                species_present = set(str(active_monsters[e][MonsterData.SPECIES_ID]) for e in entity_ids)
                if len(species_present) >= 2:
                    combats_to_run.append(entity_ids)
                
    for entity_list in combats_to_run:
        dead_ids = resolve_combat(entity_list, species_db, config, brains, epsilon, pretrained_species)
        for dead_id in dead_ids:
            death_flags[dead_id] = True

    return coexistence_rewards, death_flags
