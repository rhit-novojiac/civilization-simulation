import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import random
import csv
import threading
import torch
import argparse

# Ensure the parent directory is in the path to import correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from engine.world_gen.pipeline import WorldGenerator
from engine.world_gen.renderer import GridRenderer
from engine.state_manager import state_manager
from api.server import start_server
from engine.ecology.population import execute_first_wave, spawn_from_dens
from engine.ecology.physics import apply_macro_action, update_scent_compass
from engine.ecology.metabolism import process_metabolism
from engine.combat.resolution import resolve_combat, resolve_overworld_encounters
from ml.models.macro_net import MacroDQN
from ml.models.micro_net import MicroDQN
from ml.training.trainer import DQNTrainer, get_decayed_epsilon
from ml.inference.state_encoder import encode_macro_state
from ml.inference.action_decoder import MacroAction, MicroAction, select_macro_actions
from schema.entities import MonsterData, DenData
from schema.terrain import TileType

def load_species_db():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "species_db.json")
    with open(db_path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Civilization Ecology Simulator")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh world (deletes existing save data)")
    parser.add_argument("--log", action="store_true", help="Enable console logging (improves performance if disabled)")
    parser.add_argument("--max-ticks", type=int, default=0, help="Maximum number of ticks to run before auto-saving and exiting (0 = infinite)")
    parser.add_argument("--clear-brains", action="store_true", help="Delete all PyTorch model weights before starting")
    args = parser.parse_args()

    if not args.log:
        sys.stdout = open(os.devnull, 'w')

    print("====================================================")
    print("      Starting Civilization Ecology Simulator v1    ")
    print("====================================================")

    # 1. Load config and species database
    ConfigManager._instance = None
    config = ConfigManager()
    species_db = load_species_db()
    
    # Filter inactive species
    species_db = {k: v for k, v in species_db.items() if str(k) in config.active_species}

    print(f"World Seed      : {config.seed}")
    print(f"Dimensions      : {config.width}x{config.height}")
    print(f"Max Population  : {config.max_population}")
    print(f"Tick Delay (s)  : {config.tick_delay_seconds}")
    
    if args.fresh:
        state_file_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "active_game_state.json")
        entities_file_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "active_entities.json")
        csv_file_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "population_log.csv")
        
        for f in [state_file_tmp, entities_file_tmp, csv_file_tmp]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not remove {f} - {e}")
                    
        if config.log_world_state:
            print("\n[WorldState] --fresh flag provided. Existing save files removed.", file=sys.__stdout__)

    if args.clear_brains:
        import glob
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models")
        for f in glob.glob(os.path.join(models_dir, "*.pt")):
            basename = os.path.basename(f)
            # e.g., "species_1_macro.pt" -> parts[1] == "1"
            parts = basename.split("_")
            if len(parts) >= 2 and parts[1] in config.preserve_brains:
                continue
            
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: Could not remove {f} - {e}")
        if config.log_world_state:
            print("\n[ML] --clear-brains flag provided. All PyTorch models have been wiped.", file=sys.__stdout__)

    # 2. Launch Dashboard Server in Background Thread
    print("\n[Dashboard] Launching dashboard server...")
    server_thread = threading.Thread(target=start_server, args=(8000,), daemon=True)
    server_thread.start()
    time.sleep(1.0) # Give it a second to boot up

    # 3. Initialize / Load World Generation and State
    generator = WorldGenerator(config)
    
    # Check if a game state already exists to resume
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "active_game_state.json")
    entities_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "active_entities.json")
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "population_log.csv")
    
    if os.path.exists(state_file):
        if config.log_world_state:
            print("\n[WorldState] Found active save file. Resuming simulation...", file=sys.__stdout__)
        state_manager.load_from_disk(generator)
        
        # Purge inactive entities and dens
        state_manager.active_monsters = {
            eid: m for eid, m in state_manager.active_monsters.items()
            if str(m[MonsterData.SPECIES_ID]) in config.active_species
        }
        state_manager.dens = [
            d for d in state_manager.dens 
            if d is not None and str(d[DenData.SPECIES_ID]) in config.active_species
        ]
        
        print("\n[System] Successfully loaded map from save.", file=sys.__stdout__)
    else:
        if config.log_world_state:
            print("\n[WorldState] No save file found. Initializing fresh world...", file=sys.__stdout__)
        state_manager.initialize_world(config.seed, generator)
        biome_grid = state_manager.base_grid
        
        if config.log_world_state:
            print("[WorldState] Creating Day 1 Dens...", file=sys.__stdout__)
        from engine.world_gen.spawners import populate_dens
        state_manager.dens = populate_dens(config, biome_grid, config.seed)
        
        if config.log_world_state:
            print("[WorldState] Executing First Wave Spawning...", file=sys.__stdout__)
        execute_first_wave(0, config, species_db)
        print("\n[System] Successfully finished First Wave Spawning.", file=sys.__stdout__)
        
        # Save day 1 map renders
        img_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(img_dir, exist_ok=True)
        img_biome = GridRenderer.render(config, biome_grid, mode="Biome", altars=generator.altars)
        img_biome.save(os.path.join(img_dir, "biome_map.png"))
        img_elev = GridRenderer.render(config, biome_grid, generator.elevation, mode="Elevation")
        img_elev.save(os.path.join(img_dir, "elevation_contour.png"))
        
        state_manager.save_to_disk()

    # 4. Initialize Shared Brains (Hive Minds) for each of the 6 species
    print("\n[ML] Initializing species PyTorch Neural Networks...")
    brains = {}
    pretrained_species = set()
    for species_id in list(species_db.keys()):
        macro_policy = MacroDQN()
        macro_target = MacroDQN()
        
        micro_policy = MicroDQN()
        micro_target = MicroDQN()
        
        if config.debug_mode:
            macro_path = f"data/models/species_{species_id}_macro.pt"
            micro_path = f"data/models/species_{species_id}_micro.pt"
            loaded = False
            if os.path.exists(macro_path):
                macro_policy.load_state_dict(torch.load(macro_path))
                loaded = True
            if os.path.exists(micro_path):
                micro_policy.load_state_dict(torch.load(micro_path))
                loaded = True
            if loaded:
                pretrained_species.add(species_id)
                print(f"[ML] Loaded pretrained weights for Species {species_id}")

        macro_trainer = DQNTrainer(
            macro_policy, macro_target, 
            learning_rate=1e-4, dual_headed=True,
            sync_every=config.target_network_update_interval,
            batch_size=config.batch_size
        )
        
        micro_trainer = DQNTrainer(
            micro_policy, micro_target, 
            learning_rate=1e-4,
            sync_every=config.target_network_update_interval,
            batch_size=config.batch_size
        )
        
        brains[species_id] = {
            'macro': macro_policy,
            'macro_trainer': macro_trainer,
            'micro': micro_policy,
            'micro_trainer': micro_trainer,
            'micro_buffer': micro_trainer.memory
        }

    print("\nStarting Simulation Master Clock Loop...")
    print("Press Ctrl+C to stop and save.")
    
    delta_dens_counter = {sp: 0 for sp in list(species_db.keys())}
    
    from engine.logging.metrics import init_logs
    init_logs(is_fresh=args.fresh)
    
    try:
        while True:
            start_time = time.time()
            
            # Increment current tick
            state_manager.current_tick += 1
            tick = state_manager.current_tick
            
            # Epsilon decay step
            epsilon = get_decayed_epsilon(tick, start_epsilon=1.0, min_epsilon=0.05, decay_ticks=config.epsilon_decay_ticks)
            
            # Max Ticks Check
            if args.max_ticks > 0 and tick >= args.max_ticks:
                print(f"\n[AUTO-SAVE] Reached target of {args.max_ticks} ticks! Saving and exiting...")
                break
                
            # Extinction Check
            if len(state_manager.active_monsters) == 0 and len(state_manager.dens) == 0:
                print("\n[EXTINCTION] All monsters and dens have been destroyed! Simulation ending.")
                break
                
            # --- SECTION A: BATCHED MACRO INFERENCE & MOVEMENT ---
            update_scent_compass(state_manager, species_db)
            
            active_list = list(state_manager.active_monsters.keys())
            
            # Pre-compute spatial occupancy map to avoid O(N^2) searches inside vision grids
            occupancy_map = {}
            for mid in active_list:
                m = state_manager.active_monsters[mid]
                pos = (int(m[MonsterData.X]), int(m[MonsterData.Y]))
                if pos not in occupancy_map:
                    occupancy_map[pos] = []
                occupancy_map[pos].append(m)
                
            import numpy as np
            padded_grid = np.pad(state_manager.base_grid, 3, mode='constant', constant_values=0)
                
            # Iterate and infer actions species-by-species (Hive Mind)
            macro_actions_taken = {}  # {mid: action_idx}
            macro_states_cached = {}   # {mid: state_tensor}
            
            # Group monsters by species in a single pass
            sp_groups = {sp: [] for sp in species_db.keys()}
            for mid in active_list:
                m = state_manager.active_monsters[mid]
                if m[MonsterData.BLEEDING_TICKS] > 0:
                    m[MonsterData.BLEEDING_TICKS] -= 1
                    if m[MonsterData.BLEEDING_TICKS] == 0:
                        m[MonsterData.IS_BLEEDING] = False
                        
                if m[MonsterData.MOVEMENT_COOLDOWN] > 0:
                    m[MonsterData.MOVEMENT_COOLDOWN] -= 1
                else:
                    sp_str = str(m[MonsterData.SPECIES_ID])
                    if sp_str in sp_groups:
                        sp_groups[sp_str].append(mid)
            
            for sp_id, sp_mids in sp_groups.items():
                if not sp_mids:
                    continue
                
                # Encode states in batch
                states = []
                for mid in sp_mids:
                    state_tensor = encode_macro_state(
                        mid, 
                        state_manager.active_monsters, 
                        padded_grid,
                        occupancy_map
                    )
                    states.append(state_tensor)
                    macro_states_cached[mid] = state_tensor
                
                states_batch = torch.stack(states)
                
                # Neural inference
                with torch.no_grad():
                    q_move, q_stance = brains[sp_id]['macro'](states_batch)
                
                # Epsilon-greedy choices with Action Masking
                biomass_list = [state_manager.active_monsters[mid][MonsterData.BIOMASS] for mid in sp_mids]
                sp_threshold = species_db.get(sp_id, {}).get("reproduction_threshold", 99.0)
                diet_list = [species_db.get(sp_id, {}).get("diet", "Unknown") for _ in sp_mids]
                active_epsilon = 0.05 if sp_id in pretrained_species else epsilon
                actions = select_macro_actions(q_move, q_stance, active_epsilon, biomass_list, threshold=sp_threshold, diet_list=diet_list)
                for i, mid in enumerate(sp_mids):
                    macro_actions_taken[mid] = (actions[i][0], actions[i][1], sp_id)

            # Apply actions to world
            dens_before = len(state_manager.dens)
            macro_stances_taken = {}
            for mid, (m_act, s_act, _) in macro_actions_taken.items():
                macro_stances_taken[mid] = s_act
                apply_macro_action(mid, m_act, config, species_db)
            if len(state_manager.dens) > dens_before:
                for d in state_manager.dens[dens_before:]:
                    sp_str = str(d[2])
                    if sp_str in delta_dens_counter:
                        delta_dens_counter[sp_str] += 1
                        
            # --- SECTION A.5: CARNIVORE DEN RAIDING ---
            for mid in active_list:
                if mid not in state_manager.active_monsters:
                    continue
                m = state_manager.active_monsters[mid]
                diet = species_db.get(str(m[MonsterData.SPECIES_ID]), {}).get("diet", "")
                if diet in ("Carnivore", "Scavenger"):
                    my_x, my_y = m[MonsterData.X], m[MonsterData.Y]
                    for d_idx in range(len(state_manager.dens) - 1, -1, -1):
                        den = state_manager.dens[d_idx]
                        if den[DenData.X] == my_x and den[DenData.Y] == my_y:
                            if str(den[DenData.SPECIES_ID]) != str(m[MonsterData.SPECIES_ID]):
                                if m[MonsterData.BIOMASS] < 100.0:
                                    m[MonsterData.BIOMASS] = min(100.0, m[MonsterData.BIOMASS] + 50.0)
                                    m[MonsterData.RAIDED_DEN] = True
                                    del state_manager.dens[d_idx]
                                    break

            # --- SECTION B: COMBAT RESOLUTION ---
            coexistence_rewards, death_flags = resolve_overworld_encounters(
                state_manager.active_monsters, 
                macro_stances_taken, 
                config, 
                species_db, 
                brains, 
                epsilon,
                pretrained_species
            )
            
            # --- SECTION C: ECOLOGY METABOLISM ---
            overgrazing_penalties = process_metabolism(species_db, config)
            
            # --- SECTION D: EXPERIENCE RECORDING FOR MACRO DQN ---
            from ml.training.reward_shaper import get_macro_reward
            
            for mid, (m_act, s_act, sp_id) in macro_actions_taken.items():
                # We must record experience even if they died this tick (to learn from death)
                m_data_cached = macro_states_cached.get(mid) 
                if not m_data_cached is None:
                    current_m_data = state_manager.active_monsters.get(mid)
                    if current_m_data:
                        reward, forage_bonus = get_macro_reward(
                            current_m_data[MonsterData.BIOMASS], 
                            current_m_data[MonsterData.HP_PERCENT], 
                            action_established_den=(m_act == 5),
                            action=m_act,
                            scent_dx=current_m_data[MonsterData.SCENT_DX],
                            scent_dy=current_m_data[MonsterData.SCENT_DY],
                            species_id=sp_id,
                            species_db=species_db,
                            tracking_blood=current_m_data[MonsterData.TRACKING_BLOOD],
                            action_raided_den=current_m_data[MonsterData.RAIDED_DEN]
                        )
                        current_m_data[MonsterData.RAIDED_DEN] = False
                        # Apply herbivore foraging bonus if resting on forest/plains
                        if m_act == 4: # REST
                            spec_info = species_db.get(sp_id)
                            if spec_info and spec_info.get("diet") == "Herbivore":
                                try:
                                    tile = TileType(state_manager.get_tile(current_m_data[MonsterData.X], current_m_data[MonsterData.Y]))
                                    if tile in (TileType.FOREST, TileType.PLAINS):
                                        reward += forage_bonus
                                except Exception:
                                    pass
                    else:
                        reward = -100.0 # Death penalty
                        
                    # Add coexistence reward if applicable
                    reward += coexistence_rewards.get(mid, 0.0)
                    
                    # Add overgrazing penalty if applicable
                    reward += overgrazing_penalties.get(mid, 0.0)
                    
                    is_dead = death_flags.get(mid, False)
                    if is_dead or current_m_data is None:
                        next_state = torch.zeros_like(m_data_cached)
                    else:
                        next_state = encode_macro_state(
                            mid, 
                            state_manager.active_monsters, 
                            padded_grid, 
                            occupancy_map
                        )
                        
                    # Push transition to species macro replay buffer
                    # Action must be tuple (move_action, stance_action)
                    brains[sp_id]['macro_trainer'].push_transition(
                        m_data_cached, 
                        (m_act, s_act), 
                        reward, 
                        next_state, 
                        is_dead
                    )

            # (Metabolism was moved before Experience Recording)
            
            # Periodic den spawn cycle
            if tick % config.den_spawn_interval == 0:
                spawn_from_dens(tick, config, species_db)

            # --- SECTION E: BACKPROPAGATION & ONLINE TRAINING ---
            for sp_id in list(species_db.keys()):
                # Train Overworld Macro net
                brains[sp_id]['macro_trainer'].optimize_step()
                # Train Tactical Combat Micro net
                brains[sp_id]['micro_trainer'].optimize_step()

            # --- SECTION F: SAVE STATE & OUTPUT ---
            # Save volatile state back to json files periodically
            if tick % 10 == 0:
                state_manager.save_to_disk()
                
            # Log progress statistics to stdout
            if tick % 10 == 0 or tick == 1:
                # Calculate species stats
                species_pop = {sp: 0 for sp in list(species_db.keys())}
                for mid, m_data in state_manager.active_monsters.items():
                    species_pop[str(m_data[MonsterData.SPECIES_ID])] += 1
                    
                species_dens = {sp: 0 for sp in list(species_db.keys())}
                for d in state_manager.dens:
                    species_dens[str(d[2])] += 1
                    
                total_m = len(state_manager.active_monsters)
                total_d = len(state_manager.dens)
                agg_delta_dens = sum(delta_dens_counter.values())
                
                if config.log_clock:
                    print(f"[Clock] Tick {tick:04d} | Pop: {total_m:03d} | Dens: {total_d:02d} (+{agg_delta_dens}) | Epsilon: {epsilon:.3f} | Delay: {config.tick_delay_seconds}s")
                    
                    # Write to new CSV system
                    from engine.logging.metrics import log_population_metrics, flush_life_events
                    log_population_metrics(tick, state_manager.active_monsters, state_manager.dens, species_db, config)
                    flush_life_events()
                
                # Reset delta_dens_counter after processing it (whether logging is enabled or not)
                delta_dens_counter = {sp: 0 for sp in list(species_db.keys())}
                
            # Frame pacing
            elapsed = time.time() - start_time
            sleep_dur = max(0.005, config.tick_delay_seconds - elapsed)
            time.sleep(sleep_dur)

    except KeyboardInterrupt:
        print("\n[Clock] Simulation stopped by user. Saving final state to disk...", file=sys.__stdout__)
    finally:
        state_manager.save_to_disk()
        print("[ML] Saving PyTorch model checkpoints...", file=sys.__stdout__)
        DQNTrainer.save_models(brains, "data/models/")
        print("Goodbye!", file=sys.__stdout__)

if __name__ == "__main__":
    main()
