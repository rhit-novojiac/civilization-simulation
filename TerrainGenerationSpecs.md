# AI Coding Specification: Deterministic Terrain Generation Engine

## Project Overview

**Objective:** Build a deterministic 2D terrain generation engine and state-management system in Python using NumPy and Perlin noise.
**Goal:** Generate a realistic island map with natural slopes, gravity-based rivers, and organic biomes.
**Architecture:** The map must be completely reproducible using a single integer `seed`. Changes to the world by agents must be stored separately using a "Delta" architecture to preserve memory.

---

## 1. Configuration & Parameter Management

Do **not** hardcode any structural constants, dimensions, or generation thresholds in the Python scripts.

1. Create a `config.json` file at the root of the project to store all generation parameters.
2. Create a `ConfigManager` singleton or dataclass in Python to load and serve these parameters to the application.
3. If `seed` is set to `null` or `0` in the config, the Python engine must generate a random integer seed and log it/save it back so the user can recreate the world later.

**Expected `config.json` structure:**

```json
{
  "world": {
    "width": 1000,
    "height": 1000,
    "seed": 847294
  },
  "generation": {
    "island_noise_scale": 100.0,
    "elevation_noise_scale": 50.0,
    "river_spawn_count": 10
  },
  "biomes": {
    "seed_counts": {
      "PLAINS": 80,
      "FOREST": 60,
      "JUNGLE": 30,
      "MOUNTAIN": 25,
      "DESERT": 10
    },
    "growth_modifiers": {
      "forest_water_bonus": 2.0,
      "jungle_max_elevation": 0.6,
      "mountain_min_elevation": 0.7,
      "desert_water_penalty": 0.25,
      "desert_min_water_distance": 15,
      "desert_flatland_bonus": 2.0,
      "desert_jungle_buffer": 2
    }
  }
}
```

---

## 2. Constants & Enums

Define the following core enumerations to represent tile states in the NumPy array:

| Enum Name | Integer Value | Description |
| --- | --- | --- |
| `OCEAN` | 0 | Unwalkable border water |
| `MOUNTAIN` | 1 | High elevation terrain |
| `FOREST` | 2 | Moderate elevation, high resources |
| `JUNGLE` | 3 | High density terrain |
| `PLAINS` | 4 | Open terrain |
| `FRESHWATER` | 5 | Rivers and lakes |
| `DESERT` | 6 | Arid terrain, avoids water |

---

## 3. Terrain Generation (The 4-Pass System & Caching)

Create a `WorldGenerator` class with a `generate(seed: int) -> np.ndarray` method. 

**Seed-Stamped Terrain Caching:** 
Before running the noise generation passes, check if `data/cached_terrain_{seed}.npy` exists. If it does, load the terrain array directly from disk to bypass the heavy Perlin noise math. If it does not exist, generate the terrain as normal and then save the resulting grid to the cache file.

The generation must execute in the following four procedural passes. **Note:** Dimensions must scale dynamically based on `config.world.width` and `config.world.height`. Calculate the grid center dynamically as `(width / 2, height / 2)`.

### Pass 1: Island Mask

* **Goal:** Create a single, central landmass surrounded by ocean.
* **Logic:**
1. Calculate the distance of every `(x, y)` coordinate from the dynamic grid center.
2. Generate a low-frequency Perlin noise map (using `config.generation.island_noise_scale`) and apply it to the base radius to distort the perfect circle into jagged coastlines. The island should roughly fill 80% of the bounds.
3. If a tile falls outside the distorted radius, set `is_land = False`. Else, `is_land = True`.

### Pass 2: Elevation Topography

* **Goal:** Create a realistic slope from mountains in the center to beaches at the edges.
* **Logic:**
1. Create a radial gradient where the center tile has an elevation of `1.0` and the furthest coastline tiles have an elevation of `0.0`.
2. Generate a mid-frequency Perlin noise map (using `config.generation.elevation_noise_scale`) and add it to the radial gradient.
3. Clamp all elevation values between `0.0` and `1.0`.

### Pass 3: Hydraulic Pathfinding (Rivers & Lakes)

* **Goal:** Simulate gravity-based water flow.
* **Logic:**
1. Select `river_spawn_count` random tiles where `is_land == True` and `elevation > 0.8`.
2. For each tile, compare the elevation of its 8 neighbors. Move the "water drop" to the lowest neighbor and mark it as `FRESHWATER`.
3. **Lake Pooling:** If the water reaches a tile where all 8 neighbors are higher, mark the tile as `FRESHWATER` and raise its local elevation to match the lowest neighbor. Repeat this pooling check until the water can spill over the edge and continue downhill.
4. Terminate the river when it hits a tile where `is_land == False` (Ocean).

### Pass 4: Voronoi Biome Growth

* **Goal:** Populate the landmass with distinct biomes.
* **Logic:**
1. Spawn seed points randomly across the landmass based on `config.biomes.seed_counts`.
   - **Points of Interest (POIs):** When each initial `(x, y)` spawn coordinate is generated for a biome seed, immediately append a dictionary representing a universal Altar to the `WorldState.altars` list (e.g., `{"x": seed_x, "y": seed_y}`).
2. Constrain initial spawn points:
   - `MOUNTAIN` seeds must spawn where `elevation >= config.biomes.growth_modifiers.mountain_min_elevation`.
   - `DESERT` seeds cannot spawn within `desert_min_water_distance` tiles of any `FRESHWATER` or `OCEAN` tile. They must be strictly inland.
3. Run a vectorized NumPy BFS expansion where all biome frontiers grow outward simultaneously each round.
4. **Growth Modifiers:**
   - `FOREST` expands at `forest_water_bonus` speed when adjacent to `FRESHWATER`.
   - `JUNGLE` cannot expand into tiles with `elevation > jungle_max_elevation`.
   - `DESERT` expands at `desert_water_penalty` speed (default `0.25`, i.e. 4× slower) when expanding into tiles adjacent to `FRESHWATER`, creating a natural dry buffer around rivers and lakes.
   - `DESERT` Flatland Bonus: `DESERT` expands at `desert_flatland_bonus` speed when expanding into a tile where the absolute elevation difference is less than 0.05.
5. **Collision & Barrier Rules:**
   - Stop expansion when a growing biome collides with another or hits the Ocean/River.
   - Jungle Barrier: `DESERT` and `JUNGLE` biomes cannot touch. Their expansion must halt if they are within `desert_jungle_buffer` tiles of each other, naturally leaving `PLAINS` or `FOREST` as a buffer zone between them.
6. Any remaining unassigned land defaults to `PLAINS`.

---

## 4. Map Storage & Memory Architecture (Seed + Delta)

Do **not** store the full mapped grid array in JSON or CSV. Implement the following state management system to handle game state:

### The `WorldState` Class

* **`base_seed` (int):** The integer used to generate the pristine world.
* **`delta_map` (dict):** A JSON-serializable Python dictionary tracking only modified tiles.
* **`altars` (list):** A list of Points of Interest (POIs) such as Universal Altars spawned at biome seed locations.
* *Format Example:* `{"142,80": {"type": "Building", "owner": "Agent_1"}}`

### Data Retrieval Logic

Implement a `get_tile(x: int, y: int)` method that follows this exact hierarchy:

1. Check if the string `"x,y"` exists in `delta_map`. If yes, return the modified data object.
2. If no, query the NumPy array generated by `WorldGenerator.generate(base_seed)` and return the pristine base biome/terrain tile.

### Save/Load Operations

* **Save:** Write a tiny JSON file containing *only* the `base_seed`, the `delta_map` dictionary, and the `altars` list.
* **Load:** Read the JSON, execute `WorldGenerator.generate(base_seed)` to rebuild the base world in RAM, and load the `delta_map` and `altars` into memory to override the generated terrain where applicable.
