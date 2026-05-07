import time
from modules.PLC import TurretPLC

def sniff_plc_packets():
    # Using your current PLC IP
    plc = TurretPLC("192.168.0.101", 23000)
    if not plc.connect(): return

    print("--- RAW PLC PACKET SNIFFER ---")
    print("Format: [FN] [D1] [D2] [D3] [D4] [CRC_L] [CRC_H]")
    print("-" * 50)

    try:
        # We will send a movement command to ensure the motors are live
        # Then poll continuously to see the bytes change
        plc.send_pose(30, 0) 
        
        for i in range(15):
            # 1. Send the standard Status Poll (0xEE)
            # The pack_and_send method will receive the 7-byte response
            # We will use the socket directly to be 100% sure we see everything
            packet = bytearray([0xEE, 0x00, 0x00, 0x00, 0x00])
            crc = plc._crc16(packet)
            crc_bytes = crc.to_bytes(2, 'big')
            packet.append(crc_bytes[1])
            packet.append(crc_bytes[0])

            plc.socket_client.send(packet)
            raw_data = plc.socket_client.recv(7)

            # 2. Print Hex and Decimal for every byte
            hex_str = " ".join(f"{b:02X}" for b in raw_data)
            int_str = " ".join(f"{b:3d}" for b in raw_data)
            
            print(f"Frame {i+1:02d} | HEX: [{hex_str}]")
            print(f"         | INT: [{int_str}]")
            print("-" * 30)
            
            time.sleep(0.5)

    finally:
        plc.send_pose(0,0)
        plc.disconnect()

if __name__ == "__main__":
    sniff_plc_packets()