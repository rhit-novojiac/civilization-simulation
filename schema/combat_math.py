import math

def calculate_max_hp(endurance: int) -> int:
    return 20 + (endurance * 10)

def calculate_physical_armor(endurance: int) -> int:
    return math.floor(endurance / 2.0)

def calculate_base_damage(natural_weapon_damage: int, strength: int) -> int:
    return natural_weapon_damage + strength

def calculate_xp_required(level: int) -> int:
    return math.floor(50 * (level ** 1.5))

def calculate_xp_yield(level: int) -> int:
    return math.floor(20 * (level ** 1.2))

def calculate_active_stat(base_stat: int, level: int, stat_growth: float) -> int:
    return int(base_stat + math.floor(level * stat_growth))
