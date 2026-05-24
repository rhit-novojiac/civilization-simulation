# AI Coding Specification: DQNSpecs (PyTorch RL Architecture)

## Project Overview

**Objective:** Implement a dual-network Deep Q-Network (DQN) architecture in PyTorch for entity decision-making.
**Design Principle:** Entities use a Shared Brain ("Hive Mind") per species. The AI must process all entities of the same species in a single batched tensor to optimize compute overhead. The system uses a **Macro DQN** for Overworld navigation and a **Micro DQN** for tactical combat.

---

## 1. Network Architecture (The PyTorch Models)

Both networks must be implemented as standard Feed-Forward Neural Networks (MLPs) using `torch.nn.Linear` and `ReLU` activations.

### 1.1 The Macro DQN (Overworld Navigation)

This network processes the 7x7 vision grid and internal stats to decide where to move on the 1000x1000 map.

* **Input Layer (150 Neurons):**
* Vision Grid: 7x7 tiles $\times$ 3 features per tile (Terrain ID, Species ID, Power Ratio) = 147 inputs.
* Internal State: `hp_percent`, `biomass`, `level` = 3 inputs.
* *Note: Flatten the 7x7 grid into a 1D tensor before concatenating with internal states.*


* **Hidden Layers:** Two layers: 128 neurons $\rightarrow$ 128 neurons.
* **Output Layer (6 Neurons):** Raw Q-values for: `MOVE_N`, `MOVE_S`, `MOVE_E`, `MOVE_W`, `REST`, `ESTABLISH_DEN`.

### 1.2 The Micro DQN (Tactical Combat)

This network processes the 15x15 instance grid. To save memory, it does not use a full vision grid. It relies on relative distance vectors.

* **Input Layer (7 Neurons):**
* Internal State: `hp_percent`, `level`.
* Target Data: `delta_x`, `delta_y`, `target_hp_percent`.
* Spatial Data: `distance_to_nearest_edge` (crucial for fleeing).
* Game State: `Flee_Attempts_Remaining` (normalized 1.0 to 0.0).


* **Hidden Layers:** Two layers: 64 neurons $\rightarrow$ 64 neurons.
* **Output Layer (9 Neurons):** Raw Q-values for: 8 directional movements (N, S, E, W, NE, NW, SE, SW), `BASIC_ATTACK`.

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



---

## 4. Batched Inference Execution

Do **not** run a `for` loop to generate actions for each monster individually.

1. Collect the input state arrays for all entities of Species X.
2. **CPU Optimization:** Filter out any entity where `movement_cooldown > 0`. Do not query the neural network for these entities; simply decrement their cooldown by 1 and move on.
3. Convert the remaining entities into a single batched PyTorch tensor: `shape = (num_entities, 150)` for Macro, or `shape = (num_combatants, 7)` for Micro.
4. Pass the batch through the Policy Network.
5. Extract the `argmax` for each row to distribute the actions back to the engine.

---