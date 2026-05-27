# Specification: Abstract Auto-Battler

## 1. Core Philosophy
The 15x15 micro-grid has been deprecated. Combat is now an O(1) spatial resolution system (Abstract Auto-Battler). It prioritizes CPU efficiency for 50,000+ tick simulations by replacing pathfinding with pure deterministic and stochastic mathematics (1d20 system).

## 2. Trigger Condition (Overworld Intent Filter)
Combat is instantiated strictly when two or more entities occupy the same `(x, y)` coordinate on the Overworld map during the main engine tick, AND at least one entity is hostile.
- Herbivores default to `hostile = False`.
- Carnivores/Scavengers evaluate `hostile = True` ONLY IF their `biomass` (hunger) is `< 70.0`.
If no entities on the tile are hostile, they peacefully coexist and skip the combat loop entirely (No rewards are granted for peaceful overworld coexistence).

## 3. The Core Stat Math
Endurance no longer provides damage mitigation. Stats are strictly isolated:
- **Max HP:** `20 + (Endurance * 5)`
- **Damage Output:** `1d6 + Strength + Weapon_Damage`
- **Evasion (Armor Class):** `10 + Agility`
- **Accuracy Check:** `1d20 + Dexterity` vs Target's Evasion.
- **Initiative / Fleeing:** Governed entirely by Agility.

## 4. The Resolution Loop
When instantiated, the Overworld pauses and the `CombatInstance` executes instantly.
1. **Turn 0 (Initiator Advantage):** The entity whose movement caused the collision gets an immediate, guaranteed first action before standard Initiative applies.
2. **N-vs-N Queue:** All entities on the tile are loaded into an Initiative queue, sorted descending by `Agility`.
3. **Stamina Penalty:** If an entity attempts to `FLEE` and fails the contested Agility roll, they receive a cumulative `-2` to all future Flee rolls in this instance.
4. **Action Economy:** Both the attacker and defender(s) lose their Overworld movement for the current engine tick, flagged via `MOVEMENT_COOLDOWN = 1`.
5. **The Staredown Draw:** If ALL conscious entities in the queue output `TOLERATE` during a single round, the instance immediately ends in a bloodless draw.
6. **Flat-Footed Penalty:** If an entity outputs `TOLERATE` but receives a `BASIC_ATTACK`, their Agility is treated as `0` for the purpose of calculating their Evasion (Armor Class = Base 10) for that specific hit.

## 5. Neural Network Architecture (MicroDQN)
The AI utilizes a 12-neuron flattened input tensor utilizing **Tier Injection** to solve magnitude blindness in infinitely scaling stats.

**Input Tensor (12 Neurons):**
`[My_HP, My_STR, My_END, My_DEX, My_AGI, E_HP, E_STR, E_END, E_DEX, E_AGI, Flee_Penalty, Encounter_Tier]`
* *Relative Scaling:* All stat and HP inputs (Neurons 1-10) are divided by the absolute highest stat present in the encounter, bounding them between `0.0` and `1.0`.
* *Tier Injection:* Neuron 12 calculates the absolute scale of the fight: `min(1.0, log10(max_raw_stat) / 3.0)`.

**Output Tensor (3 Neurons):**
`[BASIC_ATTACK, FLEE, TOLERATE]`
*(Note: Output layer to be expanded to N-neurons later to support explicit party targeting).*