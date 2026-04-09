#Decision fusion - adjusts steering based on obstacle grid.

import numpy as np
from src.training import config


class DecisionFusion:
    def __init__(self):
        self.obstacle_threshold = config.SAFE_OBSTACLE_THRESHOLD
        self.direction_bias = 0.3
        self.crawl_speed = 0.15
    
    def fuse(self, nav_output, grid_flat):
        # Take nav prediction and obstacle grid, return adjusted [speed, dir].
        speed = float(nav_output[0])
        direction = float(nav_output[1])
        grid = grid_flat.reshape(3, 3)
        
        # obstacle right ahead -> brake
        center = float(grid[1, 1])
        if center > self.obstacle_threshold:
            speed = 0.0
            return [speed, direction]
        
        # something approaching (top row) -> slow down
        top_max = float(np.max(grid[0, :]))
        if top_max > self.obstacle_threshold:
            speed *= 0.5
        
        # obstacle on the left -> steer right
        left = float(grid[1, 0])
        if left > self.obstacle_threshold:
            direction += self.direction_bias
            direction = min(direction, 1.0)
        
        # obstacle on the right -> steer left
        right = float(grid[1, 2])
        if right > self.obstacle_threshold:
            direction -= self.direction_bias
            direction = max(direction, -1.0)
        
        # both sides blocked -> go slow and straight
        if left > self.obstacle_threshold and right > self.obstacle_threshold:
            direction = 0.0
            speed = min(abs(speed), self.crawl_speed)
        
        return [float(np.clip(speed, -1, 1)), float(np.clip(direction, -1, 1))]
