# AI Coding Specification: MonsterSpecs (Ecology & Growth Math)

## Project Overview

**Objective:** Define the statistical architecture, memory structures, and growth mathematics for all classless monsters in the simulation.
**Design Principle:** Individual monsters must have an extremely lightweight memory footprint. All complex math (base stats, stat growth, natural weapons) is dynamically pulled from an external JSON database. The ecology is driven by a single `Biomass` (metabolism) integer to prevent pathfinding bottlenecks while simulating hunger.

---

## 1. Entity Memory Structure (The Minimal Array)

To support thousands of monsters simultaneously, individual monster state is stored as a 1D array.

**The `active_monsters` dictionary:**

* **Key:** `entity_id` (String/UUID)
* **Value Format:** `[species_id, x, y, hp_percent, level, current_xp, biomass, age, movement_cooldown]`
* *Example:* `"1042": [2, 450, 600, 1.0, 3, 120, 45.0, 10, 0]`

All other values (Max HP, current damage, evasion) are **Derived Stats** calculated on-the-fly using the math below.

---

## 2. The Species Database (`species_db.json`)

Do **not** hardcode species templates in the Python logic. The project already contains a `species_db.json` file at the root level.

1. **Load:** The engine must read this JSON into memory at startup as `SPECIES_DB`.
2. **Read:** When calculating combat math or level-ups, query `SPECIES_DB[species_id]` to retrieve base stats, growth multipliers, and diet.
3. **Write (Mutations):** If the mutation engine generates a new species, the engine must programmatically append the new dictionary object (with a new `species_id` key) to `SPECIES_DB` and invoke `json.dump()` to permanently save it to `species_db.json`.

**Expected JSON Node Structure:**

```json
{
  "2": {
    "name": "Dire Wolf",
    "diet": "Carnivore",
    "base_stats": {"str": 8, "end": 6, "dex": 5, "agi": 6},
    "stat_growth": {"str": 2.0, "end": 1.5, "dex": 1.0, "agi": 1.0},
    "natural_weapon": {"name": "Bite", "base_damage": 8}
  }
}

```

---

## 3. The Mathematics of Stats and Leveling

When the engine needs to resolve an Overworld decision or a Micro-Grid combat turn, it calculates the monster's active stats dynamically using these formulas:

### 3.1 Stat Calculation

Stats increase linearly based on the template's Growth array. Always apply the `floor()` function to keep stats as integers.

* **Formula:** $Active\_Stat = Base\_Stat + \lfloor Level \times Stat\_Growth \rfloor$

### 3.2 Derived Combat Stats

Once the core Active Stats are calculated, calculate the Derived Stats:

* **Max HP:** $20 + (Active\_Endurance \times 5)$
* **Physical Armor:** $\lfloor Active\_Endurance \div 2 \rfloor$
* **Base Damage:** $Natural\_Weapon\_Damage + Active\_Strength$
* *Note on Current HP:* To find the entity's current HP for combat, multiply Max HP by their saved `hp_percent`.

### 3.3 Experience and Leveling Math

Monsters gain XP by surviving encounters or killing entities.

* **XP Required for Next Level:** $XP_{req} = \lfloor 50 \times Level^{1.5} \rfloor$
* **XP Yield (When Killed):** If an entity is killed, it grants the victor XP equal to: $XP_{yield} = \lfloor 20 \times Level^{1.2} \rfloor$
* **Level Up Trigger:** If `current_xp >= XP_req`, subtract `XP_req` from `current_xp`, increment `level` by 1, and set `hp_percent` to `1.0` (free heal on level up).

---

## 4. Ecology & Biomass Math (Metabolism & Reproduction)

`Biomass` functions as a combined hunger, metabolism, and reproduction tracker. This ensures monsters actively hunt/forage without requiring complex `A*` pathfinding to water sources.

### 4.1 Biomass Accumulation (Eating)

* **Herbivores (Passive Grazing):** Completely decoupled from movement actions. Herbivores automatically gain `+0.5` Biomass every tick they stand on their native biome (`PLAINS`, `FOREST`, `MOUNTAIN`, `DESERT`).
  * **Overgrazing Constraint:** If more than `10` entities occupy the same tile, the tile is overgrazed, and Herbivores receive `0.0` Biomass.
  * *Note: Biomass is strictly capped at `100.0`.*
* **Carnivores & Scavengers ("Taking a Bite"):** Gain `+2.0` Biomass per successful hit landed during combat.
* **Carnivores (Kill):** Gain Biomass dynamically based on the prey's endurance: `10.0 + (Target Endurance * 1.5)`. *(Note: If they cannibalize their own species, the total reward is divided by 4).*
* **Carnivores (Den Poaching):** Gain `+50.0` Biomass if they walk onto a Herbivore Den tile in an `AGGRESSIVE` stance, instantly destroying the Den.
* **Scavengers:** Gain `+5` Biomass every tick they stand on an Overworld tile where an entity died within the last 10 ticks.

### 4.2 Biomass Decay (Metabolism & Starvation)

* **Base Decay:** Every Overworld tick, subtract `0.1` from the entity's `biomass`.
* **Biome Modifier:** If standing on a `DESERT` tile, increase decay to `0.2` per tick to simulate harsh conditions.
* **Starvation Penalty:** If `biomass` drops to `0.0`, the entity suffers starvation. Subtract `0.10` from their `hp_percent` (10% Max HP damage) every tick until they eat or die.

### 4.3 Reproduction (Dens)

* **The Trigger:** When a monster's `biomass` reaches its `reproduction_threshold` (usually near 100), its macro DQN unlocks and heavily weights the `ESTABLISH_DEN` action.
* **The Caloric Cost:** A new entry is added to `WorldState.dens`: `[x, y, species_id, charges]`. The entity's `biomass` is violently consumed, resetting down to its base `starting_biomass` level.
* **Den Attrition:** Every 50 Overworld Ticks, each Den spawns a Level 1 entity. Dens hold a limited number of charges (default `3`). After spawning 3 entities, the Den collapses and is removed.

### 4.4 Movement Cooldowns & Terrain Affinities

* **Terrain Friction:** Entities naturally suffer a `MOVEMENT_COOLDOWN` penalty when entering rough tiles: `FOREST` (+1 tick), `MOUNTAIN` (+2 ticks), `JUNGLE` (+2 ticks).
* **Species Affinities:** Native species bypass these cooldowns entirely (e.g., Giant Spiders ignore Jungle/Forest penalties, Rock Boars ignore Mountain passes).
* **Inference Bypassing:** If an entity has a cooldown > 0, it skips AI inference completely, saving massive CPU overhead while serving as a natural "resting state".

---

## 5. System Hooks for the Neural Network

**The Vision Grid Power Level Input:**
When processing the 5x5 Macro Vision Grid, the DQN does not receive the enemy's raw stats. It receives a **Relative Power Ratio**.

* **Formula:** $Power\_Ratio = \frac{Target\_Level}{My\_Level}$
* *Behavioral Result:* If the input is `> 1.5`, the DQN naturally learns to output the `FLEE` movement vector. If the input is `< 0.8`, the DQN learns to output the `PURSUE` movement vector.

---