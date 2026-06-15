import time
from src.modules.PLC import TurretPLC
from src.modules.utils import log

def run_calibration_pass(plc, vel_value, distance=30):
    print(f"\n>>> CALIBRATING VELOCITY VALUE: {vel_value}")
    
    # 1. Reset to Zero and set speed
    plc.send_pose(0, 0)
    time.sleep(2) # Wait for homing
    plc.set_velocity(tilt_vel=1000, pan_vel=vel_value)
    
    # 2. Start the move and the timer
    input(f"Ready? Press ENTER to start move to {distance}° and start timer...")
    start_time = time.time()
    plc.send_pose(distance, 0)
    
    # 3. Manual Stop
    input("Watch the turret. Press ENTER the INSTANT it stops moving...")
    end_time = time.time()
    
    duration = end_time - start_time
    actual_speed = distance / duration
    
    print(f"RESULTS for Input {vel_value}:")
    print(f"Time Taken: {duration:.2f}s")
    print(f"Actual Speed: {actual_speed:.2f} °/s")
    
    return actual_speed

def main():
    plc = TurretPLC("192.168.0.101", 23000)
    if not plc.connect(): return

    test_values = [10, 100, 500, 1000]
    results = {}

    try:
        for val in test_values:
            actual_deg_per_sec = run_calibration_pass(plc, val)
            results[val] = actual_deg_per_sec
        
        print("\n" + "="*30)
        print("FINAL CALIBRATION TABLE")
        print(f"{'Input Value':<12} | {'Actual Speed (°/s)'}")
        print("-" * 30)
        for val, speed in results.items():
            print(f"{val:<12} | {speed:<15.2f}")
        print("="*30)

    finally:
        plc.send_pose(0, 0)
        plc.disconnect()

if __name__ == "__main__":
    main()