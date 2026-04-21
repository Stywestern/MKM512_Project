# modules/controller.py

##################################### Imports #####################################
# Libraries
from pycomm3 import LogixDriver
import numpy as np
import time

# Modules
from modules.utils import log

###################################################################################

class TurretController:
    def __init__(self, simulation=True):
        self.is_sim = simulation
        self.connected = False
        self.deadzone = 0.05
        
        # PD Parameters (Simulation & Real-time Damping)
        self.kp = 0.8
        self.kd = 0.2
        self.last_error_x = 0
        self.last_error_y = 0
        self.last_time = time.time()
        self.last_print_time = 0

        # Omron PLC parameters
        self.PLC_IP = "192.168.0.10" 
        self.tags = {
            "pan": "PC_to_PLC_PanError",   # Aiming Velocity/Position
            "tilt": "PC_to_PLC_TiltError",
            "fire": "PC_to_PLC_FireCmd"    # Laser Trigger
        }

        if self.is_sim:
            log("Controller initialized in SIMULATION mode", "INFO")
            self.connected = True
        else:
            self.connect_to_plc()

    def connect_to_plc(self):
        """Initializes the EtherNet/IP Driver for Omron NX1P2"""
        log(f"Initializing CIP Driver for Omron at {self.PLC_IP}...", "INFO")
        try:
            self.client = LogixDriver(self.PLC_IP)
            self.connected = True
            log("PLC Driver Ready", "INFO")
        except Exception as e:
            log(f"Connection Failed: {e}", "ERROR")
            self.connected = False

    def update_turret(self, target_x, target_y, dist_cm, fire_cmd):
        """
        Calculates PD response and logs real-time targeting effort.
        target_x/y: normalized error (-1.0 to 1.0)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.033

        # 1. PD Calculation (The 'Effort')
        p_x = self.kp * target_x
        p_y = self.kp * target_y 
        d_x = self.kd * (target_x - self.last_error_x) / dt
        d_y = self.kd * (target_y - self.last_error_y) / dt

        effort_x = np.clip(p_x + d_x, -1.0, 1.0)
        effort_y = np.clip(p_y + d_y, -1.0, 1.0)

        # 3. CONSOLE TELEMETRY
        if current_time - self.last_print_time >= 1.0:
            abs_err_x = int(target_x * 640)
            abs_err_y = int(target_y * 360)
            
            print(f"\n[SYSTEM STATUS - {time.strftime('%H:%M:%S')}]")
            print(f"| ERROR:  X: {abs_err_x:4d}px | Y: {abs_err_y:4d}px")
            print(f"| EFFORT: Pan: {effort_x:+.3f} | Tilt: {effort_y:+.3f}")
            print(f"| TARGET: Dist: {dist_cm:5.1f}cm | Laser: {'ACTIVE' if fire_cmd else 'OFF'}")
            print("-" * 50)
            
            self.last_print_time = current_time

        # 4. Deadzone & Transmission
        out_x = 0 if abs(effort_x) < self.deadzone else effort_x
        out_y = 0 if abs(effort_y) < self.deadzone else effort_y

        if not self.is_sim and self.connected:
            self._send_payload(out_x, out_y, fire_cmd)

        # Memory Update
        self.last_error_x = target_x
        self.last_error_y = target_y
        self.last_time = current_time

    def _send_payload(self, p, t, f):
        try:
            self.client.write(
                (self.tags["pan"], float(p)),
                (self.tags["tilt"], float(t)),
                (self.tags["fire"], bool(f))
            )
        except Exception as e:
            log(f"Data Transmission Error: {e}", "ERROR")
            self.connected = False

    def emergency_stop(self):
        if self.connected and not self.is_sim:
            self.client.write("Servo_Enable_Bit", False)
        self.update_turret(0, 0, 100, False)