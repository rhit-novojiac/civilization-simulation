import pytest
from unittest.mock import MagicMock, patch
from schema.entities import MonsterData
from engine.combat.physics import (
    calculate_active_stat,
    get_active_stats,
    get_max_hp,
    calculate_evasion,
    get_base_damage,
    get_xp_yield,
    apply_xp,
    resolve_attack
)
from engine.combat.resolution import resolve_combat, resolve_overworld_encounters
from engine.state_manager import state_manager
from ml.inference.action_decoder import MicroAction
import torch

MOCK_SPECIES_DB = {
    "1": {
        "name": "TestCarnivore",
        "diet": "Carnivore",
        "base_stats": {"str": 10, "end": 10, "dex": 10, "agi": 10},
        "stat_growth": {"str": 2.0, "end": 2.0, "dex": 2.0, "agi": 2.0},
        "natural_weapon": {"base_damage": 5}
    },
    "2": {
        "name": "TestHerbivore",
        "diet": "Herbivore",
        "base_stats": {"str": 5, "end": 15, "dex": 5, "agi": 5},
        "stat_growth": {"str": 1.0, "end": 3.0, "dex": 1.0, "agi": 1.0},
        "natural_weapon": {"base_damage": 2}
    }
}

class MockConfig:
    max_level_cap = 10
    log_combat = False
    log_ecology = False
    width = 100
    height = 100

def create_mock_monster(species_id, level=1, hp_percent=1.0, biomass=50.0):
    arr = [0] * 13
    arr[MonsterData.SPECIES_ID] = int(species_id)
    arr[MonsterData.X] = 5
    arr[MonsterData.Y] = 5
    arr[MonsterData.HP_PERCENT] = hp_percent
    arr[MonsterData.LEVEL] = level
    arr[MonsterData.CURRENT_XP] = 0
    arr[MonsterData.BIOMASS] = biomass
    arr[MonsterData.MOVEMENT_COOLDOWN] = 0
    return arr

def test_physics_calculations():
    assert calculate_active_stat(10, 2.0, 5) == 20
    monster = create_mock_monster("1", level=2)
    stats = get_active_stats(monster, MOCK_SPECIES_DB)
    assert stats["str"] == 14
    assert stats["end"] == 14
    assert get_max_hp(14) == 90
    assert calculate_evasion(14) == 24
    assert calculate_evasion(14, is_flat_footed=True) == 10
    assert get_base_damage("1", 14, MOCK_SPECIES_DB) == 19

def test_cannibalism():
    # Two Dire Wolves (Carnivores, same species)
    m1 = create_mock_monster("2", level=1)
    m2 = create_mock_monster("2", level=1)
    
    # Force low biomass so they count as "hostile"
    m1[MonsterData.BIOMASS] = 20.0
    m2[MonsterData.BIOMASS] = 20.0
    
    # Both on the same tile
    m1[MonsterData.X] = 5
    m1[MonsterData.Y] = 5
    m2[MonsterData.X] = 5
    m2[MonsterData.Y] = 5
    
    active_monsters = {"m1_id": m1, "m2_id": m2}
    stances = {"m1_id": 0, "m2_id": 0}
    
    from engine.combat.resolution import resolve_overworld_encounters
    # This should return immediately because both are species "2"
    # resulting in no combats_to_run being populated
    # If they were added, the function would crash lacking a mock `brains`
    coexistence, deaths = resolve_overworld_encounters(
        active_monsters, stances, MockConfig(), MOCK_SPECIES_DB, brains={}, epsilon=0.0
    )
    
    assert not deaths # No deaths, no combat triggered

def test_overworld_filter_non_hostile():
    # 2 non-hostile entities sharing a tile (Carnivore with high biomass, Herbivore)
    m1 = create_mock_monster("1", biomass=80.0) # High hunger -> hostile = False
    m2 = create_mock_monster("2") # Herbivore -> hostile = False
    
    state_manager.active_monsters = {"m1_id": m1, "m2_id": m2}
    state_manager.dens = []
    
    stances = {"m1_id": 0, "m2_id": 0}
    brains = {}
    config = MockConfig()
    
    # Should skip combat completely
    with patch("engine.combat.resolution.resolve_combat") as mock_rc:
        coexistence_rewards, death_flags = resolve_overworld_encounters(
            state_manager.active_monsters, stances, config, MOCK_SPECIES_DB, brains, epsilon=0.0
        )
        mock_rc.assert_not_called()

def test_overworld_filter_hostile():
    m1 = create_mock_monster("1", biomass=40.0) # Low hunger -> hostile = True
    m2 = create_mock_monster("2") 
    
    state_manager.active_monsters = {"m1_id": m1, "m2_id": m2}
    state_manager.dens = []
    stances = {"m1_id": 0, "m2_id": 0}
    
    with patch("engine.combat.resolution.resolve_combat", return_value=[]) as mock_rc:
        resolve_overworld_encounters(
            state_manager.active_monsters, stances, MockConfig(), MOCK_SPECIES_DB, {}, epsilon=0.0
        )
        mock_rc.assert_called_once()

@patch("engine.combat.resolution.random.random", return_value=1.0) # Force model inference
def test_combat_staredown_draw(mock_random):
    m1 = create_mock_monster("1", level=1)
    m2 = create_mock_monster("2", level=1)
    state_manager.active_monsters = {"m1_id": m1, "m2_id": m2}
    state_manager.combat_stats = {"kills": 0, "draws": 0, "fleds": 0}
    
    mock_model = MagicMock()
    # Output TOLERATE (idx 2)
    q_vals = torch.zeros((1, 3))
    q_vals[0, MicroAction.TOLERATE] = 10.0
    mock_model.return_value = q_vals
    
    brains = {
        "1": {"micro": mock_model, "micro_buffer": MagicMock()},
        "2": {"micro": mock_model, "micro_buffer": MagicMock()}
    }
    
    dead_ids = resolve_combat(["m1_id", "m2_id"], MOCK_SPECIES_DB, MockConfig(), brains, epsilon=0.0)
    
    assert not dead_ids
    assert m1[MonsterData.HP_PERCENT] == 1.0 # No damage taken

@patch("engine.combat.physics.random.randint")
def test_resolve_attack_flat_footed(mock_randint):
    # Mock roll so it's exactly 10.
    # Attacker Dex = 12 (Base 10 + Growth 2). Roll = 10. Total Attack = 22.
    # Defender Agi = 6 (Base 5 + Growth 1). Normal Evasion = 16.
    # If flat-footed, Evasion = 10.
    mock_randint.side_effect = [10, 3] # First for attack roll, second for damage roll
    
    attacker = create_mock_monster("1", level=1)
    defender = create_mock_monster("2", level=1)
    
    # Normally 22 > 16, so it hits either way. Let's make roll smaller.
    mock_randint.side_effect = [1, 3] # Roll=1. Total Attack = 13.
    # Evasion = 16. 13 < 16, so MISS.
    hit, dmg = resolve_attack(attacker, defender, MOCK_SPECIES_DB, is_flat_footed=False)
    assert not hit
    
    # If flat-footed, Evasion = 10. 13 >= 10, so HIT!
    mock_randint.side_effect = [1, 3] # Roll=1, DmgRoll=3
    hit, dmg = resolve_attack(attacker, defender, MOCK_SPECIES_DB, is_flat_footed=True)
    assert hit
    assert dmg > 0

@patch("engine.combat.physics.random.randint")
def test_ghost_armor_absence(mock_randint):
    """
    Asserts that damage is strictly 1d6 + STR + Wep and that Endurance 
    provides NO damage mitigation (Ghost Armor).
    """
    attacker = create_mock_monster("1", level=1)
    # Give defender massive endurance to ensure it doesn't reduce damage
    defender = create_mock_monster("2", level=1)
    defender[MonsterData.LEVEL] = 100 
    
    # 20 for attack roll (guarantee hit), 5 for damage roll
    mock_randint.side_effect = [20, 5]
    
    # Attacker STR = 12 (Base 10 + Growth 2). Base Dmg = Wep(5) + STR(12) = 17.
    # Total Damage should be 5 (dice) + 17 = 22 exactly.
    # We pass is_flat_footed=True to guarantee we hit despite the defender's level 100 AGI.
    hit, dmg = resolve_attack(attacker, defender, MOCK_SPECIES_DB, is_flat_footed=True)
    
    assert hit
    assert dmg == 22, f"Expected exactly 22 damage, but got {dmg}. Ghost armor might be present!"

@patch("engine.combat.physics.random.randint")
def test_ambush_multiplier(mock_randint):
    """
    Asserts that Carnivores get a 1.5x damage multiplier on Turn 0.
    """
    # Wolf (Carnivore) attacking Rabbit (Herbivore)
    attacker = create_mock_monster("1", level=1)  # Using "1" as it is TestCarnivore in MOCK_SPECIES_DB
    defender = create_mock_monster("2", level=1)  # TestHerbivore
    
    # Attack roll 20 (guarantee hit), Damage roll 3
    mock_randint.side_effect = [20, 3]
    
    # Attacker STR = 12. Base Dmg = Wep(5) + STR(12) = 17.
    # Total Damage = 3 (dice) + 17 = 20.
    # Ambush Multiplier (Turn 0) = floor(20 * 1.5) = 30.
    # Wait, the user said: "explicitly asserts that the final damage applied to the Rabbit's HP is exactly 28 `floor((3 + 8 + 8) * 1.5)`."
    # Let's adjust attacker's stats so STR = 8 and Weapon = 8 to match the user's math.
    # With MOCK_SPECIES_DB, "1" has Base STR 10, Growth 2. Level 1 -> STR = 12. Wep = 5.
    # To match user's exact math, let's inject a temporary species into MOCK_SPECIES_DB
    temp_db = MOCK_SPECIES_DB.copy()
    temp_db["99"] = {
        "name": "MathWolf",
        "diet": "Carnivore",
        "base_stats": {"str": 6, "end": 10, "dex": 10, "agi": 10},
        "stat_growth": {"str": 2.0, "end": 2.0, "dex": 2.0, "agi": 2.0},
        "natural_weapon": {"base_damage": 8}
    }
    attacker = create_mock_monster("99", level=1) # Active STR = 6 + 2 = 8
    
    # Base Dmg = Wep(8) + STR(8) = 16. Dice = 3. Total = 19. Wait, 19 * 1.5 = 28.5 -> floor is 28!
    
    hit, dmg = resolve_attack(attacker, defender, temp_db, is_flat_footed=False, is_turn_zero=True, terrain_val=4) # 4 = PLAINS
    
    assert hit
    assert dmg == 28, f"Expected exactly 28 damage with Turn 0 Ambush, but got {dmg}."
