# modules/controller.py

##################################### Imports #####################################
# Libraries
from pycomm3 import LogixDriver
import numpy as np
import time

# Modules
from modules.utils import log

###################################################################################

class BaseTurretController:
    """
    Abstract-style base class. 
    Handles the 'Brain' (PD math, timing, and console logging).
    """
    def __init__(self):
        # PD Parameters
        self.kp = 0.8
        self.kd = 0.2
        self.deadzone = 0.05
        
        # State Memory
        self.last_error_x = 0
        self.last_error_y = 0
        self.last_time = time.time()
        self.last_print_time = 0

    def calculate_effort(self, target_x, target_y):
        """Standard PD logic used by both Sim and Real modes."""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.033

        # PD Math
        d_x = self.kd * (target_x - self.last_error_x) / dt
        d_y = self.kd * (target_y - self.last_error_y) / dt

        effort_x = np.clip(self.kp * target_x + d_x, -1.0, 1.0)
        effort_y = np.clip(self.kp * target_y + d_y, -1.0, 1.0)

        # Update Memory
        self.last_error_x = target_x
        self.last_error_y = target_y
        self.last_time = current_time
        
        return effort_x, effort_y

    def log_status(self, target_x, target_y, effort_x, effort_y, dist_cm, fire_cmd):
        """Throttled 1Hz console print."""
        current_time = time.time()
        if current_time - self.last_print_time >= 1.0:
            print(f"\n[SYSTEM STATUS - {time.strftime('%H:%M:%S')}]")
            print(f"| ERROR:  X: {int(target_x*640):4d}px | Y: {int(target_y*360):4d}px")
            print(f"| EFFORT: Pan: {effort_x:+.3f} | Tilt: {effort_y:+.3f}")
            print(f"| TARGET: Dist: {dist_cm:5.1f}cm | Laser: {'ACTIVE' if fire_cmd else 'OFF'}")
            print("-" * 50)
            self.last_print_time = current_time

    def update_turret(self, target_x, target_y, dist_cm, fire_cmd):
        """Simulation default: Do the math and print, but don't send bytes."""
        ex, ey = self.calculate_effort(target_x, target_y)
        self.log_status(target_x, target_y, ex, ey, dist_cm, fire_cmd)
        return ex, ey


class RealTurretController(BaseTurretController):
    """
    Hardware-active class. 
    Inherits the brain, but adds the 'Body' (PLC communication).
    """
    def __init__(self, plc_ref):
        super().__init__()
        self.plc = plc_ref # Reference to TurretPLC class
        self.is_firing_latched = False
        log("Controller: PHYSICAL hardware linked", "INFO")

    def update_turret(self, target_x, target_y, dist_cm, fire_cmd):
        # 1. Use the Base class for the math
        ex, ey = super().update_turret(target_x, target_y, dist_cm, fire_cmd)

        # 2. Deadzone filter
        out_x = 0 if abs(ex) < self.deadzone else ex
        out_y = 0 if abs(ey) < self.deadzone else ey

        # 3. Physical Transmission (The PLC-specific part)
        if self.plc and self.plc.connected:
            # Scale -1.0/1.0 effort to -30/30 degrees
            # send_pose internally handles the % 360 logic
            self.plc.send_pose(out_x * 30, out_y * 20)
            
            # Handle Laser Latching
            if fire_cmd and not self.is_firing_latched:
                self.plc.set_laser(True)
                self.is_firing_latched = True
            elif not fire_cmd and self.is_firing_latched:
                self.plc.set_laser(False)
                self.is_firing_latched = False