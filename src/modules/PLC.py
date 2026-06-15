import socket
import time
from datetime import datetime
from src.modules.utils import log

class TurretPLC:
    def __init__(self, ip="192.168.0.101", port=23000):
        # Networking
        self.ip = ip
        self.port = port
        self.socket_client = None
        self.connected = False

        # Unified system state
        self.errors = {"EmergencyStop": False, "LimitSwitch": False}
        self.current_pos = {"pan": 0, "tilt": 0}

        self.current_laser_state = False
        
        self.FN_POSE = 0x01
        self.FN_VEL = 0x05
        self.FN_ACC = 0x06
        
        self.FN_SET_FIRE = 0x0B  
        self.FN_RESET_FIRE = 0x0C 

        self.FN_GET_STATUS = 0xEE

    def connect(self):
        """Initializes TCP/IP Socket connection to the Omron NX1P2"""
        try:
            self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_client.settimeout(3)
            self.socket_client.connect((self.ip, self.port))
            self.connected = True
            log(f"PLC: Connected to {self.ip}:{self.port}", "INFO")
            return True
        except socket.error as e:
            self.connected = False
            log(f"PLC Connection Error: {e}", "ERROR")
            return False

    def disconnect(self):
        if self.socket_client:
            self.socket_client.close()
            self.connected = False
            log("PLC: Connection Closed.", "INFO")

    def _crc16(self, data: bytearray):
        """The specific CRC16-Modbus algorithm required by the hardware """
        crc = 0xFFFF
        for i in range(len(data)):
            crc ^= data[i]
            for _ in range(8):
                if (crc & 0x0001) != 0:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    def _pack_and_send(self, func_code, data_payload):
        if not self.connected:
            return False, {}

        packet = bytearray([func_code]) + data_payload
        crc = self._crc16(packet)
        crc_bytes = crc.to_bytes(2, 'big')
        packet.append(crc_bytes[1])
        packet.append(crc_bytes[0])

        try:
            # 1. Send the data
            self.socket_client.send(packet)
            
            # 2. NON-BLOCKING READ: 
            self.socket_client.settimeout(0.01) 
            try:
                response = self.socket_client.recv(7)
                return self._parse_response(response)
            except socket.timeout:
                return True, {"status": "BUSY"}
        
        except socket.error as e:
            self.connected = False
            return False, {}        
            
    def send_pose(self, pan, tilt):
        """
        Sends absolute motor positions (0-359 degrees).
        """
        # Convert to 2-byte big endian unsigned integers
        p_bytes = (int(pan) % 360).to_bytes(2, 'big')
        t_bytes = (int(tilt) % 360).to_bytes(2, 'big')

        # t_bytes + p_bytes (D1,D2 = Tilt | D3,D4 = Pan)
        return self._pack_and_send(self.FN_POSE, t_bytes + p_bytes)

    def set_laser(self, state: bool):
        """Triggers the Laser Relay -K2 via function codes 0x0B/0x0C """
        fn = self.FN_SET_FIRE if state else self.FN_RESET_FIRE
        return self._pack_and_send(fn, bytearray([0,0,0,0]))
    
    def set_velocity(self, tilt_vel: int, pan_vel: int):
        """
        Sets travel speed independently for both motors.
        D1,D2 = Tilt Velocity | D3,D4 = Pan Velocity
        """
        t_v = int(tilt_vel).to_bytes(2, 'big')
        p_v = int(pan_vel).to_bytes(2, 'big')
        # Function 0x05: [0x05, T_H, T_L, P_H, P_L]
        return self._pack_and_send(0x05, t_v + p_v)

    def set_acceleration(self, tilt_acc: int, pan_acc: int):
        """
        Sets acceleration independently for both motors.
        D1,D2 = Tilt Accel | D3,D4 = Pan Accel
        """
        t_a = int(tilt_acc).to_bytes(2, 'big')
        p_a = int(pan_acc).to_bytes(2, 'big')
        # Function 0x06: [0x06, T_H, T_L, P_H, P_L]
        return self._pack_and_send(0x06, t_a + p_a)
    
    def set_dynamics(self, vel: int, acc: int, dec: int):
        """
        Sets all movement parameters at once.
        Higher values = Faster/Aggressive. Lower = Smoother.
        """
        self.set_velocity(vel)
        self.set_acceleration(acc)
        # Function 0x07 is usually Deceleration in this protocol
        val_bytes = int(dec).to_bytes(2, 'big')
        return self._pack_and_send(0x07, bytearray([0, 0]) + val_bytes)

    def poll_telemetry(self):
        """
        Sends a 'dummy' status request (0xEE) to get the latest coordinates 
        and safety flags without changing the turret's motion.
        """
        # Function 0xEE: [0xEE, 0x00, 0x00, 0x00, 0x00] + CRC
        # The data payload is just four null bytes as placeholders.
        success, telemetry = self._pack_and_send(self.FN_GET_STATUS, bytearray([0,0,0,0]))
        
        if success:
            return telemetry
        return None

    def _parse_response(self, data):
        """Deconstructs the 7-byte response into a telemetry snapshot."""
        if len(data) < 7: 
            return False, {}

        # 1. Byte Mapping (Based on PoseCtrl.py logic)
        # Byte 0: Echo of Function Code
        # Bytes 1 & 2: Axis 1 (Tilt)
        # Bytes 3 & 4: Axis 2 (Pan)
        # Bytes 5 & 6: CRC
        
        raw_tilt = (data[1] << 8) | data[2]
        raw_pan = (data[3] << 8) | data[4]

        # 2. Signed Integer Conversion (Handle -60 to +60)
        def to_signed(val):
            return val if val < 32768 else val - 65536

        telemetry = {
            "pan": to_signed(raw_pan),
            "tilt": to_signed(raw_tilt),
            "raw_bytes": data.hex(' ') # For debugging if data is weird
        }

        # Update internal error bits from the status byte (often Byte 1 in 0xEE)
        if data[0] == self.FN_GET_STATUS:
            self.errors["EmergencyStop"] = (data[2] & 0x80) != 0

        return True, telemetry
    
    def reset_to_safe(self):
        """Returns turret to (0,0) and resets dynamics to standard values."""
        if not self.connected:
            return
        
        log("PLC: Resetting hardware to safe state...", "WARNING")
        
        # 1. Reset Dynamics to 500
        self.set_velocity(tilt_vel=500, pan_vel=500)
        self.set_acceleration(tilt_acc=500, pan_acc=500)
        
        # 2. Return to Zero
        self.send_pose(pan=0, tilt=0)
        
        # 3. Ensure Laser is OFF
        self.set_laser(False)