from modules.PLC import TurretPLC

# Hardware Configuration
PLC_IP = "192.168.0.101" 
PLC_PORT = 23000

def main():
    FIRE_STATUS = False  # True = ON, False = OFF
    # -----------------------------------

    plc = TurretPLC(PLC_IP, PLC_PORT)
    
    print(f"Connecting to PLC at {PLC_IP}...")
    if not plc.connect():
        print("Connection Failed.")
        return

    try:
        action = "ON" if FIRE_STATUS else "OFF"
        print(f"Sending Command: LASER {action}")
        
        # set_laser(True) maps to 0x0B (SetF)
        # set_laser(False) maps to 0x0C (ResetF)
        success = plc.set_laser(FIRE_STATUS)

        if success:
            print(f"PLC ACK: Laser is now {action}.")
        else:
            print("PLC Error: Command not acknowledged.")

    finally:
        plc.disconnect()
        print("Script finished and connection closed.")

if __name__ == "__main__":
    main()