# modules/controller.py

##################################### Imports #####################################
# Libraries
from pycomm3 import LogixDriver

# Modules
from modules.utils import log

###################################################################################

class TurretController:
    def __init__(self, simulation=True):
        self.is_sim = simulation
        self.connected = False
        self.deadzone = 0.05
        
        # 1. Use the IP and Tag names your colleagues define in Sysmac Studio
        self.PLC_IP = "192.168.0.10" 
        self.tags = {
            "pan": "PC_to_PLC_PanError",   # Must match PLC Global Variable Name
            "tilt": "PC_to_PLC_TiltError",
            "fire": "PC_to_PLC_FireCmd"
        }

        if self.is_sim:
            log("Controller initialized in SIMULATION mode", "INFO")
            self.connected = True
        else:
            self.connect_to_plc()

    def connect_to_plc(self):
        """Initializes the EtherNet/IP Driver"""
        log(f"Initializing CIP Driver for Omron at {self.PLC_IP}...", "INFO")
        try:
            # The LogixDriver works for Omron NX/NJ over EtherNet/IP
            self.client = LogixDriver(self.PLC_IP)
            # We don't 'open' yet; pycomm3 handles connection per-write or via 'with'
            self.connected = True
            log("PLC Driver Ready (Physical connection pending first write)", "INFO")
        except Exception as e:
            log(f"Connection Failed: {e}", "ERROR")
            self.connected = False

    def update_turret(self, pan_error, tilt_error, fire_cmd):
        # 1. Apply Deadzone
        if abs(pan_error) < self.deadzone: pan_error = 0
        if abs(tilt_error) < self.deadzone: tilt_error = 0

       # Omron NX1P2 handles 'REAL' (float) types natively. 
        # We don't even need to convert to INT unless your friends prefer it!
        if self.is_sim:
            pass
        elif self.connected:
            self._send_payload(pan_error, tilt_error, fire_cmd)

    def _send_payload(self, p, t, f):
        """ The actual CIP Write operation"""
        try:
            # Efficiently write multiple tags in one 'packet'
            self.client.write(
                (self.tags["pan"], p),
                (self.tags["tilt"], t),
                (self.tags["fire"], f)
            )

        except Exception as e:
            log(f"Data Transmission Error: {e}", "ERROR")
            self.connected = False # Force reconnection attempt next cycle

    def emergency_stop(self):
        log("!!! EMERGENCY STOP: SHUTTING DOWN SERVO DRIVES !!!", "ERROR")
        if self.connected and not self.is_sim:
            # If they have an 'Enable' tag, we set it to False
            self.client.write("Servo_Enable_Bit", False)
        self.update_turret(0, 0, False)