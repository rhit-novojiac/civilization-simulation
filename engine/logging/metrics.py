import os
import csv
import glob
from collections import defaultdict
from schema.entities import MonsterData, DenData
from config import ConfigManager

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "logs")

life_events_buffer = []

def init_logs(is_fresh=False):
    """Wipes old logs if is_fresh is True, and prepares the logs directory."""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)
    elif is_fresh:
        for f in glob.glob(os.path.join(LOGS_DIR, "*.csv")):
            try:
                os.remove(f)
            except Exception:
                pass
                
    if is_fresh:
        # Also delete deprecated root-level CSVs
        data_dir = os.path.dirname(LOGS_DIR)
        for deprecated_file in ["population_log.csv", "combat_log.csv"]:
            f_path = os.path.join(data_dir, deprecated_file)
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception:
                    pass

def log_combat_outcome(tick, a_species, a_lvl, b_species, b_lvl, outcome, x, y, biome):
    """
    Appends a combat event to combat_log.csv
    Outcome is 'kill', 'flee', or 'draw'
    """
    filepath = os.path.join(LOGS_DIR, "combat_log.csv")
    file_exists = os.path.exists(filepath)
    
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Tick", "Entity A", "Level A", "Entity B", "Level B", "Outcome", "X", "Y", "Biome"])
        writer.writerow([tick, a_species, a_lvl, b_species, b_lvl, outcome, x, y, biome])

def log_life_event(tick, event_type, species_id, x, y, biome):
    """
    Buffers a life event (birth/death) to be flushed to CSV.
    """
    life_events_buffer.append([tick, event_type, species_id, x, y, biome])

def flush_life_events():
    """
    Writes the life events buffer to CSV and clears it.
    """
    if not life_events_buffer:
        return
        
    filepath = os.path.join(LOGS_DIR, "life_events.csv")
    file_exists = os.path.exists(filepath)
    
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Tick", "Event", "Species", "X", "Y", "Biome"])
            
        for event in life_events_buffer:
            writer.writerow(event)
            
    life_events_buffer.clear()

def log_population_metrics(tick, active_monsters, dens, species_db, config):
    """
    Logs population counts for every species into separate CSV files.
    """
    max_lvl = config.max_level_cap
    
    # 1. Initialize data structures for each species
    stats = {}
    for sp_id in species_db.keys():
        stats[sp_id] = {
            "pop": 0,
            "dens": 0,
            "levels": {lvl: 0 for lvl in range(1, max_lvl + 1)}
        }
        
    # 2. Count dens
    for den in dens:
        sp_id = str(den[DenData.SPECIES_ID])
        if sp_id in stats:
            stats[sp_id]["dens"] += 1
            
    # 3. Count population and levels
    for m in active_monsters.values():
        sp_id = str(m[MonsterData.SPECIES_ID])
        if sp_id in stats:
            stats[sp_id]["pop"] += 1
            lvl = m[MonsterData.LEVEL]
            lvl_bucket = min(lvl, max_lvl)
            stats[sp_id]["levels"][lvl_bucket] += 1
            
    # 4. Write to CSVs
    for sp_id, data in stats.items():
        filepath = os.path.join(LOGS_DIR, f"population_{sp_id}.csv")
        file_exists = os.path.exists(filepath)
        
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["Tick", "Population", "Dens"] + [f"Lvl{lvl}" for lvl in range(1, max_lvl + 1)]
                writer.writerow(header)
                
            row = [tick, data["pop"], data["dens"]] + [data["levels"][lvl] for lvl in range(1, max_lvl + 1)]
            writer.writerow(row)
