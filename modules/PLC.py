import socket
import time
from datetime import datetime
from modules.utils import log

class TurretPLC:
    def __init__(self, ip="192.168.0.101", port=23000):
        # Networking
        self.ip = ip
        self.port = port
        self.socket_client = None
        self.connected = False

        self.errors = {
            "MotionControl": False,
            "PLC": False,
            "EtherCAT": False,
            "EmergencyStop": False,
            "LimitSwitch": False
        }
        
        # Protocol Constants
        self.FN_POSE = 0x01
        self.FN_SET_FIRE = 0x0B
        self.FN_RESET_FIRE = 0x0C
        self.FN_HOME = 0xAA
        self.FN_GET_STATUS = 0xEE

    def connect(self):
        """Initializes TCP/IP Socket connection to the Omron NX1P2"""
        try:
            self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_client.settimeout(5.0) # Shorter timeout for better responsiveness
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
        """Formats the 7-byte packet: [FN, D1, D2, D3, D4, CRC_L, CRC_H]"""
        if not self.connected: return False

        # Construct 5-byte header
        packet = bytearray([func_code]) + data_payload
        
        # Calculate and append 2-byte CRC
        crc = self._crc16(packet)
        crc_bytes = crc.to_bytes(2, 'big')
        packet.append(crc_bytes[1]) # CRC Low
        packet.append(crc_bytes[0]) # CRC High

        try:
            self.socket_client.send(packet)
            # Receive 7-byte confirmation from PLC
            response = self.socket_client.recv(7)
            return self._parse_response(response)
        except socket.error as e:
            log(f"PLC Communication Error: {e}", "ERROR")
            self.connected = False
            return False

    def send_pose(self, pan, tilt):
        """Sends absolute motor positions (0-359 degrees) """
        # Convert to 2-byte big endian unsigned integers
        p_bytes = (int(pan) % 360).to_bytes(2, 'big')
        t_bytes = (int(tilt) % 360).to_bytes(2, 'big')
        return self._pack_and_send(self.FN_POSE, p_bytes + t_bytes)

    def set_laser(self, state: bool):
        """Triggers the Laser Relay -K2 via function codes 0x0B/0x0C """
        fn = self.FN_SET_FIRE if state else self.FN_RESET_FIRE
        return self._pack_and_send(fn, bytearray([0,0,0,0]))

    def _parse_response(self, data):
        """Deconstructs the 7-byte response to update system health """
        if len(data) < 7: return False
        
        # Check if the PLC is reporting errors (0xEE)
        if data[0] == self.FN_GET_STATUS:
            err_bits = f'{data[1]:08b}'
            safety_bits = f'{data[2]:08b}'
            
            self.errors["MotionControl"] = err_bits[7] == '1'
            self.errors["PLC"] = err_bits[6] == '1'
            self.errors["EtherCAT"] = err_bits[5] == '1'
            self.errors["LimitSwitch"] = err_bits[4] == '1'
            self.errors["EmergencyStop"] = safety_bits[7] == '1'
            
        return True