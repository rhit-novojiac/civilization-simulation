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

    print(f"World Seed      : {config.seed}")
    print(f"Dimensions      : {config.width}x{config.height}")
    print(f"Max Population  : {config.max_population}")
    print(f"Tick Delay (s)  : {config.tick_delay_seconds}")
    
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
    
    if args.fresh:
        if os.path.exists(state_file):
            os.remove(state_file)
        if os.path.exists(entities_file):
            os.remove(entities_file)
        if os.path.exists(csv_file):
            os.remove(csv_file)
        if config.log_world_state:
            print("\n[WorldState] --fresh flag provided. Existing save files removed.")

    if os.path.exists(state_file):
        if config.log_world_state:
            print("\n[WorldState] Found active save file. Resuming simulation...")
        state_manager.load_from_disk(generator)
        print("\n[System] Successfully loaded map from save.")
    else:
        if config.log_world_state:
            print("\n[WorldState] No save file found. Initializing fresh world...")
        state_manager.initialize_world(config.seed, generator)
        biome_grid = state_manager.base_grid
        
        if config.log_world_state:
            print("[WorldState] Creating Day 1 Dens...")
        from engine.world_gen.spawners import populate_dens
        state_manager.dens = populate_dens(config, biome_grid, config.seed)
        
        if config.log_world_state:
            print("[WorldState] Executing First Wave Spawning...")
        execute_first_wave(0, config, species_db)
        print("\n[System] Successfully finished First Wave Spawning.")
        
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
    for species_id in list(species_db.keys()):
        macro_policy = MacroDQN()
        macro_target = MacroDQN()
        macro_trainer = DQNTrainer(macro_policy, macro_target, learning_rate=1e-4, dual_headed=True)
        
        micro_policy = MicroDQN()
        micro_target = MicroDQN()
        micro_trainer = DQNTrainer(micro_policy, micro_target, learning_rate=1e-4)
        
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
    init_logs()
    
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
                occupancy_map[(int(m[MonsterData.X]), int(m[MonsterData.Y]))] = m
                
            # Iterate and infer actions species-by-species (Hive Mind)
            macro_actions_taken = {}  # {mid: action_idx}
            macro_states_cached = {}   # {mid: state_tensor}
            
            for sp_id in list(species_db.keys()):
                # Filter monsters of this species
                sp_mids = []
                for mid in active_list:
                    m = state_manager.active_monsters[mid]
                    if str(m[MonsterData.SPECIES_ID]) == sp_id:
                        if m[MonsterData.MOVEMENT_COOLDOWN] > 0:
                            m[MonsterData.MOVEMENT_COOLDOWN] -= 1
                        else:
                            sp_mids.append(mid)
                            
                if not sp_mids:
                    continue
                
                # Encode states in batch
                states = []
                for mid in sp_mids:
                    state_tensor = encode_macro_state(
                        mid, 
                        state_manager.active_monsters, 
                        state_manager, 
                        config.width, 
                        config.height, 
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
                actions = select_macro_actions(q_move, q_stance, epsilon, biomass_list, threshold=sp_threshold, diet_list=diet_list)
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

            # --- SECTION B: COMBAT RESOLUTION ---
            coexistence_rewards, death_flags = resolve_overworld_encounters(
                state_manager.active_monsters, 
                macro_stances_taken, 
                config, 
                species_db, 
                brains, 
                epsilon
            )
            
            # --- SECTION C: EXPERIENCE RECORDING FOR MACRO DQN ---
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
                            action_established_den=(m_act == 5)
                        )
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
                    
                    is_dead = death_flags.get(mid, False)
                    if is_dead or current_m_data is None:
                        next_state = torch.zeros_like(m_data_cached)
                    else:
                        next_state = encode_macro_state(
                            mid, 
                            state_manager.active_monsters, 
                            state_manager, 
                            config.width, 
                            config.height, 
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

            # --- SECTION D: ECOLOGY METABOLISM ---
            process_metabolism(species_db, config)
            
            # Periodic den spawn cycle
            if tick % 50 == 0:
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
        print("\n[Clock] Simulation stopped by user. Saving final state to disk...")
        state_manager.save_to_disk()
        print("Goodbye!")

if __name__ == "__main__":
    main()
