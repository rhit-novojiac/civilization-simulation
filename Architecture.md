# AI Coding Specification: Scalable Project Architecture V2

## Project Overview

**Objective:** Establish a highly modular, decoupled Python architecture using directory-based packages for a dual-scale (Macro/Micro) LitRPG simulation driven by PyTorch Deep Q-Networks.
**Design Principles:**

1. **Package-Based Scaling:** Major systems (`world_gen`, `combat`, `ml_models`) must be structured as independent directories with `__init__.py` files, rather than single monolithic files.
2. **Strict State Separation:** Game state (RAM arrays) is strictly decoupled from game logic functions.
3. **Data-Driven:** Hard math, constants, and templates are loaded from static JSON/config files.
4. **No Circular Imports:** Base data structures must live in a standalone `/schema` module.

---

## 1. Directory Structure

The repository must follow this exact hierarchy. Create these folders and their corresponding `__init__.py` files to establish the module paths.

```text
/sim_project
│
├── main.py                   # Master clock and simulation entry point
├── config.py                 # Global constants (Map size, tick rates, base seed)
│
├── /data                     # Static DBs and saved states (Single Source of Truth)
│   ├── species_db.json       # Monster template definitions
│   ├── active_game_state.json# Map seed, Dens, Altars, and modified tiles
│   └── cached_terrain_*.npy  # Fast-loading pristine terrain arrays
│
├── /schema                   # Pure data structures (No logic, prevents circular imports)
│   ├── __init__.py
│   ├── entities.py           # Dataclasses/Pydantic models for Monster arrays
│   ├── terrain.py            # Enums and Dataclasses for biomes/tiles
│   └── combat_math.py        # Shared mathematical formulas (HP, damage mitigation)
│
├── /engine                   # The deterministic game rules (No PyTorch imports allowed)
│   ├── __init__.py
│   ├── state_manager.py      # Holds RAM arrays: active_entities, dens, modified map data
│   │
│   ├── /world_gen            # Map generation and rendering pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py       # The 4-pass Voronoi/Perlin controller
│   │   ├── biome_logic.py    # Voronoi and growth modifiers
│   │   ├── renderer.py       # Outputting map visualizations to /images
│   │   └── spawners.py       # Day 1 Den scattering logic
│   │
│   ├── /combat               # 15x15 Micro-grid resolution
│   │   ├── __init__.py
│   │   ├── grid_builder.py   # Extracts local terrain from the macro map
│   │   ├── physics.py        # Hit chances, stat comparisons, movement limits
│   │   └── resolution.py     # Win/loss conditions, XP allocation, instance cleanup
│   │
│   └── /ecology              # Macro-world background simulation
│       ├── __init__.py
│       ├── metabolism.py     # Biomass decay, starvation HP loss
│       └── population.py     # Den creation, spawning logic, mutation rolls
│
├── /ml                       # The Neural Network infrastructure (PyTorch strictly contained here)
│   ├── __init__.py
│   │
│   ├── /models               # The PyTorch Architectures
│   │   ├── __init__.py
│   │   ├── base_mlp.py       # Shared utility classes for feed-forward nets
│   │   ├── macro_net.py      # Overworld 5x5 vision grid network
│   │   └── micro_net.py      # Tactical 15x15 combat network
│   │
│   ├── /inference            # Translating Engine data -> AI decisions
│   │   ├── __init__.py
│   │   ├── state_encoder.py  # Converts Python arrays into PyTorch Tensors
│   │   └── action_decoder.py # Converts NN output indexes back to Engine enums/actions
│   │
│   └── /training             # The Learning Loop
│       ├── __init__.py
│       ├── trainer.py        # Loss calculation and backprop
│       ├── replay_buffer.py  # Experience memory management
│       └── reward_shaper.py  # Mathematical curves for dynamic rewards (HP/Biomass shaping)
│
└── /api                      # Placeholder for future external connections
    ├── __init__.py
    └── server.py             # Websocket/FastAPI for broadcasting state

```

---

## 2. Module Responsibilities & Boundaries

### `/schema` (The Data Foundation)

* **Role:** Holds all `dataclasses`, `Enums`, and `Pydantic` models.
* **Rule:** This folder cannot import from `/engine` or `/ml`. Both `/engine` and `/ml` will import from here.

### `/engine/state_manager.py` (The Memory Core)

* **Role:** The only file allowed to hold the live data arrays in RAM.
* **Contains:** `active_monsters` dict, `WorldState.dens` list, and the Delta Map tracker.
* **Rule:** Other modules may read from here, but only specific execution functions (like `combat/resolution.py` concluding a fight) are allowed to overwrite values.

### `/engine/ecology` (The Overworld)

* **Role:** Handles background survival mechanics every Overworld Tick.
* **Rule:** Iterates through all entities, applies Biomass decay via `metabolism.py`, and checks entity coordinates to see if a collision/combat instance is triggered.

### `/engine/combat` (The Micro-Grid)

* **Role:** Handles the collision of two entities.
* **Rule:** Generates the temporary 15x15 grid, queries `/ml/inference` for micro-actions, calculates the stat math, and outputs the survivor/rewards back to `state_manager.py` and `/ml/training`.

### `/ml/inference` (The Bridge)

* **Role:** Translates engine arrays into AI decisions.
* **Rule:** `/engine` modules do not know how PyTorch works. They simply pass arrays to `inference`. This module handles the tensor conversions, runs the batched forward pass, and returns Python integers back to the engine.

### `/ml/training` (The Learner)

* **Role:** Runs invisibly in the background.
* **Rule:** Every time the engine resolves an action, it pushes the `(state, action, reward, next_state, done)` tuple into the Replay Buffer. At the end of a tick, `trainer.py` runs `train_step()` to update the PyTorch weights.

---

## 3. The Master Loop (`main.py`)

To prevent spaghetti code, the main clock loop must be linear and strictly batched.

```python
# Conceptual flow for main.py
def game_loop():
    # 1. Start of Tick (Overworld = 1 In-Game Minute)
    current_states = state_manager.get_all_entity_states()
    
    # 2. Get AI Decisions (Batched by Species)
    # *Note: Skip Neural Network inference for any entity with movement_cooldown > 0*
    actions = inference.get_macro_actions(current_states)
    
    # 3. Apply Actions & Physics
    rewards, collisions = ecology.apply_macro_actions(actions)
    
    # 4. Resolve Micro-Combat (if any collisions occurred)
    for encounter in collisions:
        combat_rewards = combat.resolve_encounter(encounter)
        trainer.store_micro_memories(combat_rewards)
    
    # 5. Store Macro Memories & Train
    trainer.store_macro_memories(rewards)
    trainer.train_step()
    
    # 6. End of Tick (Advance clock, process Biomass decay/Spawns)
    ecology.process_end_of_tick()

```