import math
from schema.entities import MonsterData
from schema.terrain import TileType
from config import ConfigManager

def calculate_active_stat(base_stat, growth, level):
    """
    Active_Stat = Base_Stat + floor(Level * Stat_Growth)
    """
    return int(base_stat + math.floor(level * growth))

def get_active_stats(monster, species_db):
    """
    Computes all active stats for a monster.
    """
    species_id = str(monster[MonsterData.SPECIES_ID])
    level = monster[MonsterData.LEVEL]
    
    spec_info = species_db.get(species_id, {})
    base_stats = spec_info.get("base_stats", {"str": 1, "end": 1, "dex": 1, "agi": 1})
    stat_growth = spec_info.get("stat_growth", {"str": 0, "end": 0, "dex": 0, "agi": 0})
    
    active_str = calculate_active_stat(base_stats.get("str", 1), stat_growth.get("str", 0.0), level)
    active_end = calculate_active_stat(base_stats.get("end", 1), stat_growth.get("end", 0.0), level)
    active_dex = calculate_active_stat(base_stats.get("dex", 1), stat_growth.get("dex", 0.0), level)
    active_agi = calculate_active_stat(base_stats.get("agi", 1), stat_growth.get("agi", 0.0), level)
    
    return {
        "str": active_str,
        "end": active_end,
        "dex": active_dex,
        "agi": active_agi
    }

def get_max_hp(active_endurance):
    """
    Max HP: 20 + (Active_Endurance * 5)
    """
    return 20 + (active_endurance * 5)

def calculate_evasion(active_agility, is_flat_footed=False):
    """
    Armor Class (Evasion): 10 + Agility (or 0 if flat-footed)
    """
    agi = 0 if is_flat_footed else active_agility
    return 10 + agi

def get_base_damage(species_id, active_strength, species_db):
    """
    Base Damage Component = Natural_Weapon_Damage + Active_Strength
    """
    spec_info = species_db.get(str(species_id), {})
    natural_weapon = spec_info.get("natural_weapon", {"base_damage": 1})
    natural_weapon_dmg = natural_weapon.get("base_damage", 1)
    return natural_weapon_dmg + active_strength

def get_xp_requirement(level):
    """
    XP Required for Next Level: 50
    """
    return 50

def get_xp_yield(level):
    """
    XP Yield (When Killed): level * 100
    """
    return level * 100

def apply_xp(monster, xp_gained):
    """
    Gains XP and processes level-up checks.
    Returns True if a level-up occurred.
    """
    level = monster[MonsterData.LEVEL]
    current_xp = monster[MonsterData.CURRENT_XP]
    
    current_xp += xp_gained
    leveled_up = False
    
    while True:
        if level >= ConfigManager().max_level_cap:
            current_xp = get_xp_requirement(level)  # Max out XP bar at level cap
            break
            
        xp_req = get_xp_requirement(level)
        if current_xp >= xp_req:
            current_xp -= xp_req
            level += 1
            monster[MonsterData.HP_PERCENT] = 1.0  # Heal to full on level up
            leveled_up = True
        else:
            break
            
    monster[MonsterData.LEVEL] = level
    monster[MonsterData.CURRENT_XP] = current_xp
    return leveled_up

import random
def resolve_attack(attacker, defender, species_db, is_flat_footed=False, is_turn_zero=False, terrain_val=None):
    """
    Resolves a basic attack from attacker onto defender using 1d20 system.
    Modifies defender's hp_percent.
    Returns (hit_success, damage_dealt)
    """
    att_species_id = attacker[MonsterData.SPECIES_ID]
    
    att_stats = get_active_stats(attacker, species_db)
    def_stats = get_active_stats(defender, species_db)
    
    # Accuracy Check: 1d20 + DEX
    attack_roll = random.randint(1, 20) + att_stats["dex"]
    evasion = calculate_evasion(def_stats["agi"], is_flat_footed)
    
    if attack_roll >= evasion:
        # Damage Output: 1d6 + STR + Weapon_Damage
        base_dmg = get_base_damage(att_species_id, att_stats["str"], species_db)
        damage_dealt = random.randint(1, 6) + base_dmg
        
        # Turn 0 Ambush Multiplier
        if is_turn_zero:
            att_diet = species_db.get(str(att_species_id), {}).get("diet")
            def_diet = species_db.get(str(defender[MonsterData.SPECIES_ID]), {}).get("diet")
            if att_diet == "Carnivore" and def_diet == "Herbivore":
                multiplier = 1.5
                if str(att_species_id) == "5" and terrain_val is not None:
                    try:
                        terrain_enum = TileType(int(terrain_val))
                        if terrain_enum in (TileType.JUNGLE, TileType.FOREST):
                            multiplier = 2.0
                    except ValueError:
                        pass
                damage_dealt = math.floor(damage_dealt * multiplier)
                
        def_max_hp = get_max_hp(def_stats["end"])
        
        # Apply damage as reduction in hp_percent
        hp_percent_reduction = float(damage_dealt) / float(def_max_hp)
        defender[MonsterData.HP_PERCENT] = max(0.0, defender[MonsterData.HP_PERCENT] - hp_percent_reduction)
        
        return True, damage_dealt
    else:
        return False, 0
