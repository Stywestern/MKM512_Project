import time
import numpy as np
from modules.PLC import TurretPLC
from modules.utils import log

def run_manual_overwatch():
    # 1. Initialize and Connect
    plc = TurretPLC("192.168.0.101", 23000)
    
    log("Connecting to Omron Turret...", "INFO")
    if not plc.connect():
        return

    try:
        # 2. Setup Dynamics (Increase timeout to 30s to prevent the crash you saw)
        plc.socket_client.settimeout(30.0) 
        
        # Set slow sweeping dynamics
        plc.set_velocity(tilt_vel=500, pan_vel=10)
        plc.set_acceleration(tilt_acc=500, pan_acc=100)
        
        log("Overwatch Active. Press Ctrl+C to Stop and Reset.", "INFO")

        while True:
            # Move to +60 (Python waits here for the move to finish)
            log("Sweeping to +60°", "INFO")
            success, _ = plc.send_pose(60, 0)
            if not success: break

            # Move to -60 (Python waits here for the move to finish)
            log("Sweeping to -60°", "INFO")
            success, _ = plc.send_pose(-60, 0)
            if not success: break

    except KeyboardInterrupt:
        log("\nUser interrupted script.", "WARNING")
    
    except Exception as e:
        log(f"Script Error: {e}", "ERROR")

    finally:
        # 3. THE EXIT FUNCTION: Always happens regardless of how it stops
        log("Executing Safety Exit...", "INFO")
        try:
            # Return to center at a normal speed
            plc.set_velocity(500, 500)
            plc.send_pose(0, 0)
            plc.disconnect()
            log("Turret Homing. Connection Closed.", "SUCCESS")
        except:
            log("Could not perform safe exit (Socket already closed).", "ERROR")

if __name__ == "__main__":
    run_manual_overwatch()