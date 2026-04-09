#Safe stop controller - kills the car if something looks wrong.

import numpy as np
from src.training import config


class SafeStopController:
    def __init__(self):
        self.consecutive_failures = 0
    
    def check(self, frame, fused_output, confidence=1.0, grid=None):
        # Check all safety conditions. Returns [0,0] if we should stop.
        speed, direction = fused_output[0], fused_output[1]
        
        mean_brightness = float(np.mean(frame))
        
        # too dark - camera failed?
        if mean_brightness < config.SAFE_DARK_THRESHOLD:
            print(f"SAFE STOP: Dark frame (mean={mean_brightness:.0f})")
            return [0.0, 0.0]
        
        # too bright - glare/whiteout
        if mean_brightness > config.SAFE_GLARE_THRESHOLD:
            print(f"SAFE STOP: Overexposed (mean={mean_brightness:.0f})")
            return [0.0, 0.0]
        
        # frozen camera - same image over and over
        if float(np.var(frame)) < config.SAFE_VARIANCE_THRESHOLD:
            print(f"SAFE STOP: Frozen camera (var={np.var(frame):.1f})")
            return [0.0, 0.0]
        
        # model is confused
        if confidence < config.SAFE_MIN_CONFIDENCE:
            self.consecutive_failures += 1
            if self.consecutive_failures >= config.SAFE_MAX_FAILURES:
                print(f"SAFE STOP: Low confidence ({confidence:.2f}) for {self.consecutive_failures} frames")
                return [0.0, 0.0]
        else:
            self.consecutive_failures = 0
        
        # obstacle grid says stop but nav says go
        if grid is not None:
            center = float(grid.reshape(3, 3)[1, 1])
            if center > config.SAFE_OBSTACLE_THRESHOLD and abs(speed) > 0.1:
                print(f"SAFE STOP: Obstacle override (center={center:.2f})")
                return [0.0, direction]
        
        return [speed, direction]
