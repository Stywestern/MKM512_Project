# modules/comms.py

##################################### Imports #####################################
# Libraries

# Modules
from modules.utils import log

###################################################################################

class TurretController:
    def __init__(self, simulation=True):
        self.is_sim = simulation
        self.connected = False
        self.deadzone = 0.05
        
        # PLC parameters (Example addresses)
        self.PLC_IP = "192.168.0.10"
        self.REG_PAN = 40001
        self.REG_TILT = 40002
        self.REG_FIRE = 40003

        if self.is_sim:
            log("Controller initialized in SIMULATION mode", "INFO")
            self.connected = True
        else:
            self.connect_to_plc()

    def connect_to_plc(self):
        """Placeholder for Modbus TCP or Ethernet/IP connection logic"""
        log(f"Attempting to connect to PLC at {self.PLC_IP}...", "INFO")

        # Example: self.client = ModbusClient(self.PLC_IP)
        self.connected = False # Default to false until hardware is ready

    def update_turret(self, pan_error, tilt_error, fire_cmd):
        # 1. Apply Deadzone
        if abs(pan_error) < self.deadzone: pan_error = 0
        if abs(tilt_error) < self.deadzone: tilt_error = 0

        # 2. Scale for PLC (Modbus registers often expect 0-65535 or -32768 to 32767)
        # We'll map -1.0...1.0 to -1000...1000
        pan_int = int(pan_error * 1000)
        tilt_int = int(tilt_error * 1000)
        fire_int = 1 if fire_cmd else 0

        if self.is_sim:
            # log(f"SIM: Pan {pan_int} | Tilt {tilt_int} | Fire {fire_int}", "DEBUG")
            pass
        elif self.connected:
            self._send_payload(pan_int, tilt_int, fire_int)

    def _send_payload(self, p, t, f):
        # This is where the actual Modbus/Ethernet code will go
        # self.client.write_registers(self.REG_PAN, [p, t, f])
        pass

    def emergency_stop(self):
        """Safety override to kill all motor movement"""
        log("!!! EMERGENCY STOP SENT TO PLC !!!", "ERROR")
        self.update_turret(0, 0, False)