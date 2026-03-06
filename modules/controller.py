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
        """
        Main entry point for sending data.
        pan/tilt_error: float -1.0 to 1.0
        fire_cmd: boolean
        """
        
        # 1. Scale for PLC (Convert float to Integer)
        # PLC registers usually handle integers. 0.5 becomes 500.
        pan_int = int(pan_error * 1000)
        tilt_int = int(tilt_error * 1000)
        fire_int = 1 if fire_cmd else 0

        if self.is_sim:
            # In simulation, we just log the 'packets' being sent
            # Only log occasionally to avoid spamming the console
            pass 
        else:
            if self.connected:
                self._send_payload(pan_int, tilt_int, fire_int)

    def _send_payload(self, p, t, f):
        """The actual low-level network write"""
        # self.client.write_registers(self.REG_PAN, [p, t, f])
        pass

    def emergency_stop(self):
        """Safety override to kill all motor movement"""
        log("!!! EMERGENCY STOP SENT TO PLC !!!", "ERROR")
        self.update_turret(0, 0, False)