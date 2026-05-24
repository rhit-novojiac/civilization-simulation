from enum import IntEnum

class TileType(IntEnum):
    OCEAN = 0
    MOUNTAIN = 1
    FOREST = 2
    JUNGLE = 3
    PLAINS = 4
    FRESHWATER = 5
    DESERT = 6

def is_walkable(tile_type: TileType) -> bool:
    """
    Determines if a tile is walkable by entities.
    Ocean and Freshwater are not walkable.
    """
    if tile_type in (TileType.OCEAN, TileType.FRESHWATER):
        return False
    return True
