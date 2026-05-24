import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_cdt
import random
from typing import List, Tuple
from config import ConfigManager
from schema.terrain import TileType


def generate_perlin_noise_2d(width: int, height: int, scale: float, seed: int) -> np.ndarray:
    """
    Vectorized, fully deterministic 2D Perlin Noise generator in pure NumPy.
    Ensures 100% platform-independent results without binary dependency issues.
    """
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=int)
    rng.shuffle(p)
    p = np.stack([p, p]).flatten()

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x = x / scale
    y = y / scale

    x0 = x.astype(int)
    y0 = y.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    tx = x - x0
    ty = y - y0

    fade_x = tx * tx * tx * (tx * (tx * 6 - 15) + 10)
    fade_y = ty * ty * ty * (ty * (ty * 6 - 15) + 10)

    def hash_coords(xi, yi):
        return p[(p[xi % 256] + yi) % 256]

    g00 = hash_coords(x0, y0) % 8
    g10 = hash_coords(x1, y0) % 8
    g01 = hash_coords(x0, y1) % 8
    g11 = hash_coords(x1, y1) % 8

    gradients = np.array([
        [1, 1], [-1, 1], [1, -1], [-1, -1],
        [1, 0], [-1, 0], [0, 1], [0, -1]
    ], dtype=np.float32)
    lengths = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients = gradients / lengths

    grad00 = gradients[g00]
    grad10 = gradients[g10]
    grad01 = gradients[g01]
    grad11 = gradients[g11]

    dot00 = tx * grad00[..., 0] + ty * grad00[..., 1]
    dot10 = (tx - 1) * grad10[..., 0] + ty * grad10[..., 1]
    dot01 = tx * grad01[..., 0] + (ty - 1) * grad01[..., 1]
    dot11 = (tx - 1) * grad11[..., 0] + (ty - 1) * grad11[..., 1]

    nx0 = dot00 + fade_x * (dot10 - dot00)
    nx1 = dot01 + fade_x * (dot11 - dot01)
    noise = nx0 + fade_y * (nx1 - nx0)

    return noise * np.sqrt(2)


# 3x3 cross struct for 8-neighbor dilation
_STRUCT_8 = np.ones((3, 3), dtype=bool)


class WorldGenerator:
    def __init__(self, config: ConfigManager = None):
        self.config = config if config is not None else ConfigManager()
        self.elevation = None
        self.is_land = None
        self.is_river = None
        self.initial_seeds = None
        self.altars = []

    def _get_8_neighbors(self, y: int, x: int, height: int, width: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbors.append((ny, nx))
        return neighbors

    def _get_flatland_mask(self, frontier_mask: np.ndarray) -> np.ndarray:
        height, width = frontier_mask.shape
        flatland_mask = np.zeros((height, width), dtype=bool)
        for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            y_start, y_end = max(0, -dy), min(height, height - dy)
            x_start, x_end = max(0, -dx), min(width, width - dx)
            
            ny_start, ny_end = max(0, dy), min(height, height + dy)
            nx_start, nx_end = max(0, dx), min(width, width + dx)
            
            # Direct slice operations - no large temporary array allocations!
            flatland_mask[y_start:y_end, x_start:x_end] |= (
                frontier_mask[ny_start:ny_end, nx_start:nx_end] &
                (np.abs(self.elevation[y_start:y_end, x_start:x_end] - self.elevation[ny_start:ny_end, nx_start:nx_end]) < 0.05)
            )
        return flatland_mask

    def _dilate_mask(self, mask: np.ndarray, steps: int) -> np.ndarray:
        result = mask.copy()
        for _ in range(steps):
            result = binary_dilation(result, _STRUCT_8)
        return result

    def generate(self, seed: int) -> np.ndarray:
        """
        Executes the 4-Pass Terrain Generation system.
        Returns a 2D numpy uint8 array of TileType values.
        """
        import os
        cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", f"cached_terrain_{seed}.npy")
        cache_elev = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", f"cached_elev_{seed}.npy")
        cache_altars = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", f"cached_altars_{seed}.npy")
        
        if os.path.exists(cache_file) and os.path.exists(cache_elev) and os.path.exists(cache_altars):
            print(f"[WorldState] Loading cached terrain for seed {seed}...")
            self.elevation = np.load(cache_elev)
            biome_grid = np.load(cache_file)
            self.altars = np.load(cache_altars).tolist()
            return biome_grid

        self.altars = []
        width = self.config.width
        height = self.config.height
        cx, cy = width / 2.0, height / 2.0

        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)

        # ==========================================
        # PASS 1: Island Mask
        # ==========================================
        base_radius = 0.8 * min(width, height) / 2.0

        island_noise = generate_perlin_noise_2d(
            width, height, self.config.island_noise_scale, seed
        )
        distorted_radius = base_radius * (1.0 + island_noise * 0.3)
        self.is_land = dist_from_center <= distorted_radius

        margin_y = int(height * 0.02)
        margin_x = int(width * 0.02)
        self.is_land[:margin_y, :] = False
        self.is_land[height - margin_y:, :] = False
        self.is_land[:, :margin_x] = False
        self.is_land[:, width - margin_x:] = False

        # ==========================================
        # PASS 2: Elevation Topography
        # ==========================================
        radial_gradient = np.zeros((height, width), dtype=np.float32)
        land_mask = self.is_land
        radial_gradient[land_mask] = np.clip(
            1.0 - (dist_from_center[land_mask] / distorted_radius[land_mask]),
            0.0, 1.0
        )

        elev_noise = generate_perlin_noise_2d(
            width, height, self.config.elevation_noise_scale, seed + 1
        )
        self.elevation = np.clip(radial_gradient + elev_noise * 0.35, 0.0, 1.0)
        self.elevation[~self.is_land] = 0.0

        # ==========================================
        # PASS 3: Hydraulic Pathfinding (Rivers & Lakes)
        # ==========================================
        self.is_river = np.zeros((height, width), dtype=bool)

        candidates = np.argwhere(self.is_land & (self.elevation > 0.8))
        if len(candidates) < self.config.river_spawn_count:
            flat_land = np.argwhere(self.is_land)
            if len(flat_land) > 0:
                order = np.argsort(self.elevation[flat_land[:, 0], flat_land[:, 1]])[::-1]
                candidates = flat_land[order[:max(self.config.river_spawn_count, len(flat_land))]]

        rng = np.random.default_rng(seed + 2)
        if len(candidates) > 0:
            chosen_indices = rng.choice(
                len(candidates),
                size=min(self.config.river_spawn_count, len(candidates)),
                replace=False
            )
            river_spawns = [tuple(candidates[i]) for i in chosen_indices]
        else:
            river_spawns = []

        for spawn in river_spawns:
            curr = spawn
            self.is_river[curr[0], curr[1]] = True
            river_visited = {curr}

            for _ in range(1500):
                y, x = curr
                if not self.is_land[y, x]:
                    break

                neighbors = self._get_8_neighbors(y, x, height, width)
                if not neighbors:
                    break

                unvisited = [n for n in neighbors if n not in river_visited]
                pool = unvisited if unvisited else neighbors
                lowest = min(pool, key=lambda n: self.elevation[n[0], n[1]])
                ly, lx = lowest
                le = self.elevation[ly, lx]

                if le >= self.elevation[y, x]:
                    self.elevation[y, x] = le + 1e-5

                curr = lowest
                river_visited.add(curr)
                self.is_river[ly, lx] = True

        # ==========================================
        # PASS 4: Voronoi Biome Growth (NumPy BFS)
        # ==========================================
        # Initialize: OCEAN and FRESHWATER are pre-assigned; land = 255 (unassigned)
        biome_grid = np.full((height, width), 255, dtype=np.uint8)
        biome_grid[~self.is_land] = TileType.OCEAN
        biome_grid[self.is_river] = TileType.FRESHWATER

        dry_land = self.is_land & ~self.is_river

        # Precompute freshwater adjacency via dilation (vectorized, instant)
        is_adj_freshwater = binary_dilation(self.is_river, structure=_STRUCT_8) & ~self.is_river

        # --- Place biome seeds ---
        seed_rng = np.random.default_rng(seed + 3)

        def place_seeds(biome_id: TileType, count: int, candidates_mask: np.ndarray):
            coords = np.argwhere(candidates_mask & (biome_grid == 255))
            if len(coords) == 0:
                coords = np.argwhere(dry_land & (biome_grid == 255))
            if len(coords) == 0:
                return
            chosen = seed_rng.choice(
                len(coords), size=min(count, len(coords)), replace=False
            )
            for idx in chosen:
                ry, rx = coords[idx]
                biome_grid[ry, rx] = biome_id
                self.altars.append([int(rx), int(ry)])

        # MOUNTAIN seeds constrained to high elevation
        mountain_mask = dry_land & (self.elevation >= self.config.mountain_min_elevation)
        if mountain_mask.sum() < self.config.seed_counts.get("MOUNTAIN", 25):
            mountain_mask = dry_land  # fallback
        place_seeds(TileType.MOUNTAIN, self.config.seed_counts.get("MOUNTAIN", 25), mountain_mask)

        # JUNGLE seeds constrained to low-mid elevation
        jungle_mask = dry_land & (self.elevation <= self.config.jungle_max_elevation)
        place_seeds(TileType.JUNGLE, self.config.seed_counts.get("JUNGLE", 30), jungle_mask)

        # FOREST seeds — prefer near rivers but anywhere is fine
        place_seeds(TileType.FOREST, self.config.seed_counts.get("FOREST", 60), dry_land)

        # PLAINS seeds — open land
        place_seeds(TileType.PLAINS, self.config.seed_counts.get("PLAINS", 80), dry_land)

        # DESERT seeds — inland, avoiding freshwater/ocean, and avoiding JUNGLE seeds
        dist_from_water = distance_transform_cdt(dry_land, metric='chessboard')
        jungle_seeds = biome_grid == TileType.JUNGLE
        forbidden_by_jungle = self._dilate_mask(jungle_seeds, self.config.desert_jungle_buffer)
        
        desert_mask = dry_land & (dist_from_water >= self.config.desert_min_water_distance) & ~forbidden_by_jungle
        if desert_mask.sum() < self.config.seed_counts.get("DESERT", 10):
            max_dist = dist_from_water.max()
            desert_mask = dry_land & (dist_from_water >= max(1, max_dist - 2)) & ~forbidden_by_jungle
        place_seeds(TileType.DESERT, self.config.seed_counts.get("DESERT", 10), desert_mask)

        self.initial_seeds = biome_grid.copy()

        # --- NumPy Multi-Label BFS Expansion ---
        # We maintain one boolean "frontier" mask per biome.
        # Each round we dilate all frontiers simultaneously; 
        # first-come-first-served for contested cells.
        # FOREST gets extra expansion steps near freshwater (forest_water_bonus).

        biome_ids = [TileType.MOUNTAIN, TileType.JUNGLE, TileType.FOREST, TileType.PLAINS, TileType.DESERT]
        forest_bonus = max(1, int(round(self.config.forest_water_bonus)))
        desert_penalty = self.config.desert_water_penalty  # speed multiplier near water (<1)

        # Build per-biome seed masks
        frontiers = {b: (biome_grid == b) for b in biome_ids}

        max_rounds = width + height  # enough to fill the whole island
        jungle_buffer = self.config.desert_jungle_buffer
        for round_idx in range(max_rounds):
            unassigned = biome_grid == 255

            # Early exit: nothing left to fill
            if not unassigned.any():
                break

            # Forbidden zones for JUNGLE and DESERT to avoid collision buffer violations
            existing_desert = biome_grid == TileType.DESERT
            existing_jungle = biome_grid == TileType.JUNGLE
            forbidden_for_jungle = self._dilate_mask(existing_desert, jungle_buffer)
            forbidden_for_desert = self._dilate_mask(existing_jungle, jungle_buffer)

            # Compute candidate expansions per biome
            expansions = {}
            for b_id in biome_ids:
                if not frontiers[b_id].any():
                    continue

                # Base dilation (all biomes expand 1 step)
                dilated = binary_dilation(frontiers[b_id], structure=_STRUCT_8)

                # For FOREST near freshwater: extra dilation steps = bonus
                if b_id == TileType.FOREST and forest_bonus > 1:
                    forest_frontier_near_water = frontiers[b_id] & is_adj_freshwater
                    if forest_frontier_near_water.any():
                        for _ in range(forest_bonus - 1):
                            forest_frontier_near_water = binary_dilation(
                                forest_frontier_near_water, structure=_STRUCT_8
                            )
                        dilated = dilated | forest_frontier_near_water

                # Constraint: JUNGLE cannot expand into high elevation
                if b_id == TileType.JUNGLE:
                    dilated = dilated & (self.elevation <= self.config.jungle_max_elevation)
                    dilated = dilated & ~forbidden_for_jungle

                # Constraint: DESERT expands slower when adjacent to freshwater.
                if b_id == TileType.DESERT:
                    dilated = dilated & ~forbidden_for_desert

                    # Flatland bonus expansion
                    flatland_bonus = max(1, int(round(self.config.desert_flatland_bonus)))
                    if flatland_bonus > 1:
                        flatland_mask = self._get_flatland_mask(frontiers[b_id])
                        flatland_candidates = dilated & flatland_mask & unassigned & dry_land
                        
                        if flatland_candidates.any():
                            dilated_2 = binary_dilation(flatland_candidates, structure=_STRUCT_8)
                            flatland_mask_2 = self._get_flatland_mask(flatland_candidates)
                            flatland_candidates_2 = dilated_2 & flatland_mask_2 & unassigned & dry_land
                            
                            # JUNGLE barrier also applies to 2nd-step flatland candidates!
                            flatland_candidates_2 = flatland_candidates_2 & ~forbidden_for_desert
                            dilated = dilated | flatland_candidates_2

                    # Cells that would expand INTO a tile adjacent to freshwater are penalised
                    desert_near_water = dilated & is_adj_freshwater
                    desert_away_water = dilated & ~is_adj_freshwater
                    # penalty=0.25 → only expand near-water every 4th round
                    skip = max(1, int(round(1.0 / desert_penalty)))
                    if (round_idx % skip) != 0:
                        dilated = desert_away_water  # suppress near-water expansion this round
                    else:
                        dilated = desert_near_water | desert_away_water

                # Only target unassigned dry land
                expansions[b_id] = dilated & unassigned & dry_land

            # Apply expansions — iterate biomes in order; first claim wins
            claimed_this_round = np.zeros((height, width), dtype=bool)
            for b_id in biome_ids:
                if b_id not in expansions:
                    continue
                new_cells = expansions[b_id] & ~claimed_this_round
                biome_grid[new_cells] = b_id
                frontiers[b_id] = new_cells  # only newly claimed cells are the new frontier
                claimed_this_round |= new_cells

            # If nothing was claimed this round, we're done
            if not claimed_this_round.any():
                break

        # Any remaining unassigned land (blocked by constraints e.g. high-elev jungle) -> PLAINS
        biome_grid[biome_grid == 255] = TileType.PLAINS

        np.save(cache_elev, self.elevation)
        np.save(cache_file, biome_grid)
        np.save(cache_altars, np.array(self.altars))

        return biome_grid
