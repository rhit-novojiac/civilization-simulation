import numpy as np
from PIL import Image
from config import ConfigManager
from schema.terrain import TileType

class GridRenderer:
    @staticmethod
    def render(config: ConfigManager, biome_grid: np.ndarray, elevation_grid: np.ndarray = None, mode="Biome", altars=None):
        """
        Renders the generated 2D grid into a PIL Image based on the selected mode.
        Modes: "Biome", "Elevation"
        """
        height, width = biome_grid.shape
        
        # Base RGB array before upscaling
        base_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if mode == "Biome":
            # Map TileType integer values to RGB colors from config
            for b_id, color in config.biome_colors.items():
                mask = biome_grid == b_id
                base_img[mask] = color
                
            # Draw altars
            if altars is not None:
                for altar in altars:
                    ax, ay = altar[0], altar[1]
                    # Draw a bright magenta 3x3 cross to make them stand out
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                        ny, nx = ay + dy, ax + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            base_img[ny, nx] = [255, 0, 255]
                
        elif mode == "Elevation" and elevation_grid is not None:
            import matplotlib.pyplot as plt
            import io
            
            fig, ax = plt.subplots(
                figsize=(width / 100.0, height / 100.0), 
                dpi=100 * config.pixels_per_cell
            )
            # Create a terrain contour plot
            ax.contour(elevation_grid, levels=15, cmap='terrain')
            ax.set_aspect('equal')
            ax.axis('off')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close(fig)
            
            return Image.open(buf)
        else:
            # Greyscale elevation fallback
            if elevation_grid is not None:
                grey = (elevation_grid * 255).astype(np.uint8)
                base_img[..., 0] = grey
                base_img[..., 1] = grey
                base_img[..., 2] = grey
            else:
                raise ValueError(f"Unknown render mode or missing elevation grid for: {mode}")

        # Scale up using NumPy repeat (nearest neighbor scaling) for pixel perfect grids
        if config.pixels_per_cell > 1:
            scaled_img = np.repeat(base_img, config.pixels_per_cell, axis=0)
            scaled_img = np.repeat(scaled_img, config.pixels_per_cell, axis=1)
        else:
            scaled_img = base_img
            
        return Image.fromarray(scaled_img)
