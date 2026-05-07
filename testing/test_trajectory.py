import time
import threading
from modules.PLC import TurretPLC

# Hardware Configuration
PLC_IP = "192.168.0.101" # 
PLC_PORT = 23000          # 

def run_stepped_test():
    plc = TurretPLC(PLC_IP, PLC_PORT)
    if not plc.connect(): return

    try:
        # We define the range in standard degrees
        test_steps = list(range(-60, 70, 1))
        
        for step_deg in test_steps:
            print(f"\n[STEP] Requesting: {step_deg}°")
            
            success = plc.send_pose(step_deg, 0)
            
            if success:
                print(f"Status: Acknowledged. Moving to {step_deg}...")
            
            #time.sleep(1)

        plc.send_pose(0, 0)

    finally:
        plc.disconnect()


if __name__ == "__main__":
    run_stepped_test()