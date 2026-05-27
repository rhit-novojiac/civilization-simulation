# AI Coding Specification: DQNSpecs (PyTorch RL Architecture)

## Project Overview

**Objective:** Implement a dual-network Deep Q-Network (DQN) architecture in PyTorch for entity decision-making.
**Design Principle:** Entities use a Shared Brain ("Hive Mind") per species. The AI must process all entities of the same species in a single batched tensor to optimize compute overhead. The system uses a **Macro DQN** for Overworld navigation and a **Micro DQN** for tactical combat.

---

## 1. Network Architecture (The PyTorch Models)

Both networks must be implemented as standard Feed-Forward Neural Networks (MLPs) using `torch.nn.Linear` and `ReLU` activations.

### 1.1 The Macro DQN (Overworld Navigation)

This network processes the 7x7 vision grid and internal stats to decide where to move on the 1000x1000 map.

* **Input Layer (152 Neurons):**
* Vision Grid: 7x7 tiles $\times$ 3 features per tile (Terrain ID, Species ID, Power Ratio) = 147 inputs.
* Internal State: `hp_percent`, `biomass`, `level` = 3 inputs.
* Scent Compass (O(1)): `scent_dx`, `scent_dy` = 2 inputs. The O(1) scent compass provides global tracking vectors to preys or predators without pathfinding overhead.
* *Note: Flatten the 7x7 grid into a 1D tensor before concatenating with internal states and scent compass.*


* **Output Layer (8 Neurons Dual-Headed):**
  * **Movement Head (6 Neurons):** Raw Q-values for: `MOVE_N`, `MOVE_S`, `MOVE_E`, `MOVE_W`, `REST`, `ESTABLISH_DEN`.
  * **Stance Head (2 Neurons):** Raw Q-values for: `AGGRESSIVE`, `PEACEFUL`.

### 1.2 The Micro DQN (Tactical Combat)

This network utilizes an **Abstract Auto-Battler** system, abandoning spatial grids in favor of a 12-neuron Tier Injected flat tensor.

* **Input Layer (12 Neurons):**
* `[My_HP, My_STR, My_END, My_DEX, My_AGI, E_HP, E_STR, E_END, E_DEX, E_AGI, Flee_Penalty, Encounter_Tier]`
* All HP/Stats are divided by the absolute highest stat in the encounter.
* `Encounter_Tier` = `min(1.0, log10(max_raw_stat) / 3.0)`

* **Hidden Layers:** Two layers: 64 neurons $\rightarrow$ 64 neurons.
* **Output Layer (3 Neurons):** Raw Q-values for: `[BASIC_ATTACK, FLEE, TOLERATE]`.

---

## 2. Dynamic Reward Shaping (Q-Learning Targets)

The agent must optimize the Bellman equation: $Q(s, a) = r + \gamma \max_{a'} Q(s', a')$
Do not use static flats for all rewards. Implement state-dependent shaping so the AI dynamically alters its priorities based on its physiological needs.

### 2.1 Macro Reward Table

* **Time Penalty:** `-0.01` per tick (forces the entity to seek resources).
* **Starvation:** `-1.0` per tick while `biomass == 0.0`.
* **Eat/Forage (Hunger Curve):** Reward scales inversely with current fullness. A starving monster gets high points; a full monster gets near zero.
* *Formula:* $+10.0 \times \frac{100.0 - current\_biomass}{100.0}$


* **Reproduce:** `+50.0` (Triggered on successful `ESTABLISH_DEN`).
* **Lethal Mistake:** `-100.0` (If the entity dies in a micro-combat instance, pass this penalty back to the Overworld state that triggered the encounter).

### 2.2 Micro Reward Table

* **Turn Penalty:** `-0.1` per turn (encourages ending the fight quickly).
* **Deal Damage:** Scaling reward based on the impact of the hit.
* *Formula:* $+10.0 \times \frac{Damage\_Dealt}{Target\_Max\_HP}$


* **Take Damage:** Scaling penalty based on severity.
* *Formula:* $-10.0 \times \frac{Damage\_Taken}{My\_Max\_HP}$


* **Kill Target:** `+50.0` (Combat Victory).
* **Flee/Escape (Survival Curve):** Reward scales inversely with current HP. Escaping at full health is worth nothing, but escaping at death's door is highly rewarded.
* *Formula:* $+50.0 \times (1.0 - current\_hp\_percent)$

* **Staredown Draw:** `+5.0` (If combat ends in a `TOLERATE` standoff, both entities are rewarded slightly for avoiding lethal conflict).



---

## 3. Hyperparameters & Memory (The Training Loop)

To ensure stable learning and prevent memory leaks, implement a standard Experience Replay Buffer and a Target Network.

* **Experience Replay Buffer:** Maximum capacity of `50,000` transitions. Stored as `(state, action, reward, next_state, done)`.
* **Batch Size:** `128`. (Process training updates in batches to optimize compute overhead).
* **Gamma ($\gamma$):** `0.99` (Discount factor for future rewards).
* **Target Network Update:** Copy Policy Net weights to Target Net every `1,000` steps.
* **Epsilon-Greedy Exploration:**
* Start Epsilon: `1.0` (100% random actions).
* Min Epsilon: `0.05` (5% random actions).
* Decay: Linear decay over the first `100,000` overworld ticks.



## 4. Model Checkpointing (Pretraining)

To facilitate rapid iteration during short debug simulations, the `DQNTrainer` supports PyTorch model checkpointing. 
- Models are saved as `.pt` files to `data/models/` during shutdown.
- When `debug_mode` is enabled in `config.py`, the simulation will attempt to load these pretrained weights on startup.
- If a species successfully loads pretrained weights, its exploration rate (`epsilon`) is overridden to a permanent `0.05`, bypassing the standard decay curve.

---

## 5. Batched Inference Execution

Do **not** run a `for` loop to generate actions for each monster individually.

1. Collect the input state arrays for all entities of Species X.
2. **CPU Optimization:** Filter out any entity where `movement_cooldown > 0`. Do not query the neural network for these entities; simply decrement their cooldown by 1 and move on.
3. Convert the remaining entities into a single batched PyTorch tensor: `shape = (num_entities, 152)` for Macro, or `shape = (num_combatants, 12)` for Micro.
4. Pass the batch through the Policy Network.
5. Extract the `argmax` for each row to distribute the actions back to the engine.

---

## 5. CPU Performance & Vectorization Protocols

To prevent Python loop iteration and Garbage Collector overhead from severely bottlenecking high-population frames:
1. **Vision Grid Extraction:** Avoid nested Python loops with individual `get_tile()` method calls. Instead, pad the terrain array (`numpy.pad`) once per tick, and use highly optimized native NumPy 2D array indexing/slicing (`padded_grid[my:my+7, mx:mx+7]`) for immediate extraction.
2. **PyTorch Tensor Masking:** Epsilon-Greedy action masking (e.g., preventing Herbivores from picking the `AGGRESSIVE` stance) MUST NOT dynamically allocate memory via `torch.tensor(python_list)` inside the hot loop. Instead, allocate static global tensors at boot (`_biomass_mask`, `_diet_mask`) and apply Boolean tensor masking over the entire batch instantly via C/C++ backend.

---