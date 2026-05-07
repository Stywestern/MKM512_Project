from modules.PLC import TurretPLC 

# Hardware Configuration from the Project Schematic
PLC_IP = "192.168.0.101" 
PLC_PORT = 23000

def run_motion_test(pan_deg, tilt_deg):
    # 1. Initialize our new PLC object
    plc = TurretPLC(PLC_IP, PLC_PORT)
    
    # 2. Connect (Handshake)
    if not plc.connect():
        print("FAILED to connect to Omron PLC.")
        return

    try:
        print(f"--- Sending Motion Command ---")
        print(f"Target Pan (L/R): {pan_deg}°")
        print(f"Target Tilt (U/D): {tilt_deg}°")

        # 3. Send Pose (This handles the 7-byte packing and CRC16)
        # Internally: [0x01, Pan_H, Pan_L, Tilt_H, Tilt_L, CRC_L, CRC_H]
        success = plc.send_pose(pan_deg, tilt_deg)

        if success:
            print("SUCCESS: Command received and acknowledged by PLC.")
            # Check for hardware flags in the response
            if plc.errors["EmergencyStop"]:
                print("WARNING: Emergency Stop is currently ACTIVE on hardware!")
            if plc.errors["LimitSwitch"]:
                print("WARNING: Axis has hit a physical Limit Switch!")
        else:
            print("ERROR: PLC rejected the packet or CRC failed.")

    except Exception as e:
        print(f"Script Error: {e}")
    
    finally:
        # 4. Clean Disconnect
        plc.disconnect()
        print("Connection closed.")

if __name__ == "__main__":
    # For both directions, the range of degrees is [-60, 60], if you try to give more, like 80, PLC will not register the command
    TEST_PAN = 0  # positive is to Turret's Left
    TEST_TILT = 0  # positive is to Turret's Down
    
    run_motion_test(TEST_PAN, TEST_TILT)