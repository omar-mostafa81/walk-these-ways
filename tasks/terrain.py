####################
# Terrain generator
####################
import numpy as np

from isaacgym.terrain_utils import *

class Terrain:
    """Class to handle terrain creation, usually a bunch of subterrains surrounding by a flat border area.

    Subterrains are spread on a given number of columns and rows. Each column might correspond to a given
    type of subterrains (slope, stairs, ...) and each row to a given difficulty (from easiest to hardest).
    """

    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return

        meshified_terrains = False
        if not meshified_terrains:
            # Create subterrains on the fly based on Isaac subterrain primitives
            
            # Retrieving proportions of each kind of subterrains
            keys = list(cfg["terrainProportions"].keys())
            vals = list(cfg["terrainProportions"].values())
            self.terrain_keys = []
            self.terrain_proportions = []
            sum = 0.0
            for key, val in zip(keys, vals):
                if val != 0.0:
                    sum += val
                    self.terrain_keys.append(key)
                    self.terrain_proportions.append(sum)

            self.horizontal_scale = 0.1  # Resolution of the terrain height map
            self.border_size = 8.0  # Size of the flat border area all around the terrains
            self.env_length = cfg["mapLength"]  # Length of subterrains
            self.env_width = cfg["mapWidth"]  # Width of subterrains
            self.env_rows = cfg["numLevels"]
            self.env_cols = cfg["numTerrains"]
            self.num_maps = self.env_rows * self.env_cols
            self.num_per_env = int(num_robots / self.num_maps)
            self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

            # Number of height map cells for each subterrain in width and length
            self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
            self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

            self.border = int(self.border_size / self.horizontal_scale)
            self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
            self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border
            self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.float32)

            self.vertical_scale = 0.005
            if cfg["curriculum"]:
                self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
            else:
                self.randomized_terrain()
            self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
            self.heightsamples = self.height_field_raw
        else:
            # Load high resolution and optimized subterrains
            self.vertical_scale = 1.0
            from glob import glob
            terrain_datas = glob("terrain_data/*.npz")
            terrain_datas.sort()
            data = np.load(terrain_datas[-1], allow_pickle=True)
            self.vertices = data["vertices"].astype(np.float32)
            self.triangles = data["triangles"].astype(np.uint32)
            self.terrain_params = data["params"].item()

            # Terrain parameters
            self.horizontal_scale = 1.0 / float(self.terrain_params["resolution"])
            self.env_length = self.terrain_params["terrainSize"]
            self.env_width = self.terrain_params["terrainSize"]
            self.border_size = self.terrain_params["borderSize"]
            self.env_rows = self.terrain_params["numLevels"]
            self.env_cols = self.terrain_params["numTerrains"]
            self.num_maps = self.env_rows * self.env_cols
            self.num_per_env = int(num_robots / self.num_maps)
            self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

            self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
            self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

            self.border = int(self.border_size/self.horizontal_scale)
            self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
            self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border
            self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.float32)

            self.heightsamples = data["heightmap"].astype(np.float32)
            # self.border_size = 0 # We already take into account the offset

            for i in range(self.env_rows):
                for j in range(self.env_cols):
                    env_origin_x = (i + 0.5) * self.env_length
                    env_origin_y = (j + 0.5) * self.env_width
                    x1 = int((env_origin_x - 0.5) / self.horizontal_scale) + self.border
                    x2 = int((env_origin_x + 0.5) / self.horizontal_scale) + self.border
                    y1 = int((env_origin_y - 0.5) / self.horizontal_scale) + self.border
                    y2 = int((env_origin_y + 0.5) / self.horizontal_scale) + self.border
                    env_origin_z = np.max(self.heightsamples[x1:x2, y1:y2])*self.vertical_scale
                    self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def randomized_terrain(self):
        """Spawn random subterrain without ordering them by type or difficulty
        according to their rows and columns"""

        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        """Spawn subterrain ordering them by type (one type per column)
        and by difficulty (first row is easiest, last one is hardest)
        """

        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    width=self.width_per_env_pixels,
                                    length=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.6
                step_height = 0.05 + 0.115 * difficulty

                k = 0
                while k < len(self.terrain_proportions) and choice >= self.terrain_proportions[k]:
                    k += 1
                if k == len(self.terrain_proportions):
                    # The sum of terrain proportions is not >= 1
                    # Defaulting to flat ground
                    continue

                if self.terrain_keys[k] == "pyramid_sloped_upwards":
                    pyramid_sloped_terrain(
                        terrain, slope=slope, platform_size=2.0
                    )
                elif self.terrain_keys[k] == "pyramid_sloped_downwards":
                    pyramid_sloped_terrain(
                        terrain, slope=-slope, platform_size=2.0
                    )
                elif self.terrain_keys[k] == "pyramid_stairs_upwards":
                    pyramid_stairs_terrain(
                        terrain, step_width=1.0, step_height=step_height, platform_size=3.0
                    )
                elif self.terrain_keys[k] == "pyramid_stairs_downwards":
                    pyramid_stairs_terrain(
                        terrain,
                        step_width=1.0,
                        step_height=-step_height,
                        platform_size=3.0,
                    )
                elif self.terrain_keys[k] == "pyramid_stairs_downwards_small":
                    pyramid_stairs_terrain(
                        terrain,
                        step_width=0.4,
                        step_height=-step_height*0.5,
                        platform_size=2.0,
                    )
                elif self.terrain_keys[k] == "discrete_obstacles":
                    #discrete_obstacles_height = 0.025 + difficulty * 0.15
                    discrete_obstacles_height = 0.05 + difficulty * 0.195
                    terrain_noise_magnitude = 0.01 + 0.02 * difficulty

                    rectangle_min_size = 0.5
                    rectangle_max_size = 2.0
                    num_rectangles = 80
                    discrete_obstacles_terrain(
                        terrain,
                        discrete_obstacles_height,
                        rectangle_min_size,
                        rectangle_max_size,
                        num_rectangles,
                        platform_size=1.5,
                    )
                    random_uniform_terrain(
                        terrain,
                        min_height=-terrain_noise_magnitude,
                        max_height=terrain_noise_magnitude,
                        step=0.005,
                        downsampled_scale=0.075,
                    )

                elif self.terrain_keys[k] == "stepping_stones":
                    stepping_stones_size = 2 - 1.8 * difficulty
                    stone_distance = 0.1
                    stepping_stones_terrain(
                        terrain,
                        stone_size=stepping_stones_size,
                        stone_distance=stone_distance,
                        max_height=0.0,
                        platform_size=2.0,
                    )
                elif self.terrain_keys[k] == "random_uniform":
                    terrain_noise_magnitude = 0.10 * difficulty
                    random_uniform_terrain(
                        terrain,
                        min_height=-terrain_noise_magnitude,
                        max_height=terrain_noise_magnitude,
                        step=0.01,
                        downsampled_scale=0.2,
                    )
                elif self.terrain_keys[k] == "rough_sloped_upwards":
                    terrain_noise_magnitude = 0.01 + 0.02 * difficulty
                    pyramid_sloped_terrain(
                        terrain, slope=slope, platform_size=2.0
                    )
                    random_uniform_terrain(
                        terrain,
                        min_height=-terrain_noise_magnitude,
                        max_height=terrain_noise_magnitude,
                        step=0.005,
                        downsampled_scale=0.075,
                    )
                elif self.terrain_keys[k] == "rough_sloped_downwards":
                    terrain_noise_magnitude = 0.01 + 0.02 * difficulty
                    pyramid_sloped_terrain(
                        terrain, slope=-slope, platform_size=2.0
                    )
                    random_uniform_terrain(
                        terrain,
                        min_height=-terrain_noise_magnitude,
                        max_height=terrain_noise_magnitude,
                        step=0.005,
                        downsampled_scale=0.075,
                    )
                elif self.terrain_keys[k] == "rough_stairs_upwards":
                    terrain_noise_magnitude = 0.01 + 0.02 * difficulty
                    #step_height = 0.05 + 0.195 * difficulty
                    #step_height = 0.05 + 0.245 * difficulty
                    step_height = 0.05 + 0.295 * difficulty
                    pyramid_stairs_terrain(
                        terrain,
                        step_width=0.8, #0.35,
                        step_height=step_height,
                        platform_size=2.0
                    )
                    random_uniform_terrain(
                        terrain,
                        min_height=-terrain_noise_magnitude,
                        max_height=terrain_noise_magnitude,
                        step=0.005,
                        downsampled_scale=0.075,
                    )

                elif self.terrain_keys[k] == "rough_stairs_downwards":
                    terrain_noise_magnitude = 0.01 + 0.02 * difficulty
                    #step_height = 0.05 + 0.195 * difficulty
                    #step_height = 0.05 + 0.245 * difficulty
                    step_height = 0.05 + 0.295 * difficulty
                    pyramid_stairs_terrain(
                        terrain,
                        step_width=0.8, #0.35,
                        step_height=-step_height,
                        platform_size=2.0,
                    )
                    random_uniform_terrain(
                        terrain,
                        min_height=-terrain_noise_magnitude,
                        max_height=terrain_noise_magnitude,
                        step=0.005,
                        downsampled_scale=0.075,
                    )
                elif self.terrain_keys[k] == "gap":
                    terrain_noise_magnitude = 0.01 + 0.02 * difficulty
                    gap_size = 0.05 + 0.25 * difficulty
                    height_noise = 0.01 + 0.05 * difficulty
                    self.gap_terrain(
                        terrain,
                        step_size=0.8,
                        gap_size=gap_size,
                        height_noise=height_noise,
                        platform_size=1.0,
                    )
                    random_uniform_terrain(
                        terrain,
                        min_height=-terrain_noise_magnitude,
                        max_height=terrain_noise_magnitude,
                        step=0.005,
                        downsampled_scale=0.075,
                    )

                else:
                    # Flat ground
                    pass

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2. - 0.5) / self.horizontal_scale)
                x2 = int((self.env_length / 2. + 0.5) / self.horizontal_scale)
                y1 = int((self.env_width / 2. - 0.5) / self.horizontal_scale)
                y2 = int((self.env_width / 2. + 0.5) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def gap_terrain(terrain, step_size, gap_size, height_noise=0.02, platform_size=1.):
        """Generate a subterrain with a gap the robot must cross without falling in.

        Args:
            terrain (Subterrain class): subterrain being built
            gap_size (float): width of the gap [meters]
            height_noise (float): amplitude of the random noise added to the terrain [meters]
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        """

        # Convert parameters to discrete units
        step_size = int(step_size / terrain.horizontal_scale)
        gap_size = int(gap_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)
        fall_max = int(-0.15/terrain.vertical_scale)
        fall_min = int(-0.35/terrain.vertical_scale)
        fall_range = np.arange(fall_min, fall_max, step=1)
        height_noise = int(height_noise / terrain.vertical_scale)
        height_range = np.arange(-height_noise, height_noise, step=1)

        # Build the gap
        start = step_size # - half_gap_size
        stop = terrain.width - step_size  # + half_gap_size
        while (stop - start) > platform_size:
            fall = np.random.choice(fall_range)
            height = np.random.choice(height_range)
            terrain.height_field_raw[start-gap_size: stop+gap_size, start-gap_size: stop+gap_size] = fall
            terrain.height_field_raw[start:stop, start:stop] = height
            start += step_size
            stop -= step_size


