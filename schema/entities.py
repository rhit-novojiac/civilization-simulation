from enum import IntEnum

class MonsterData(IntEnum):
    SPECIES_ID = 0
    X = 1
    Y = 2
    HP_PERCENT = 3
    LEVEL = 4
    CURRENT_XP = 5
    BIOMASS = 6
    AGE = 7
    MOVEMENT_COOLDOWN = 8
    HAS_ACTIVE_DEN = 9
    SCENT_UPDATE_TIMER = 10
    SCENT_DX = 11
    SCENT_DY = 12
    IS_BLEEDING = 13
    BLEEDING_TICKS = 14
    TRACKING_BLOOD = 15
    RAIDED_DEN = 16

class DenData(IntEnum):
    X = 0
    Y = 1
    SPECIES_ID = 2
    CHARGES = 3
    CREATOR_ID = 4
