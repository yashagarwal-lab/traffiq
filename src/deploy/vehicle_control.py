#Vehicle control - sends speed/direction to motors.


class VehicleController:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.current_speed = 0.0
        self.current_direction = 0.0
        
        if not dry_run:
            self._init_hardware()
    
    def _init_hardware(self):
        # TODO: set up actual motor driver when we get the car
        # probably PCA9685 servo driver or GPIO PWM
        pass
    
    def send(self, speed, direction):
        speed = max(-1.0, min(1.0, speed))
        direction = max(-1.0, min(1.0, direction))
        self.current_speed = speed
        self.current_direction = direction
        
        if self.dry_run:
            return
        
        # TODO: convert [-1,1] to PWM values
        # steering_pulse = 1500 + direction * 500
        # throttle_pulse = 1500 + speed * 500
        pass
    
    def emergency_stop(self):
        self.send(0.0, 0.0)
    
    def cleanup(self):
        self.emergency_stop()
