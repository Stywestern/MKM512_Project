# src/modules/controller.py

##################################### Imports #####################################
# Standart Libraries
import numpy as np
import time
import threading
import queue
import math
import os
import csv

# Modules
from src.modules.utils import log
import src.config as config

###################################################################################

######################################################################################################################################################################
#                                                                       SIM CONTROLLER (Digital Twin)
######################################################################################################################################################################

class SimTurretController:
    """
    Standalone Virtual Turret. Used if PLC is not connected. 
    Calculates physical offsets and simulates motor accumulation in the console.
    """
    def __init__(self):
        self.target_csv_counter = 0
        self.is_firing_latched = False

        # 1. Unified PD Parameters 
        self.kp = 0.06  
        self.kd = 0.02  
        self.deadzone_deg = 0.5 
        
        # 2. Mathematical Memory
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_time = time.time()

        # 3. Virtual Kinematic State
        self.current_pan = 0.0  
        self.current_tilt = 0.0
        self.overwatch_dir = "left"

        # 4. Virtual Camera Physical Properties
        self.cam_res_x = config.FRAME_WIDTH
        self.cam_res_y = config.FRAME_HEIGHT
        self.fov_pan = 60.0   
        self.fov_tilt = 40.0  

        self.deg_per_pixel_x = self.fov_pan / self.cam_res_x
        self.deg_per_pixel_y = self.fov_tilt / self.cam_res_y

        # 5. Parallax constants
        self.laser_offset_y_cm = 4.4 
        self.laser_offset_x_cm = -5.5 

    def calculate_effort(self, target_deg_x, target_deg_y):
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Deadzone calculation
        if dt <= 0.001: dt = 0.033
        if abs(target_deg_x) < self.deadzone_deg: target_deg_x = 0.0
        if abs(target_deg_y) < self.deadzone_deg: target_deg_y = 0.0

        # Position calculation
        d_x = self.kd * (target_deg_x - self.last_error_x) / dt
        d_y = self.kd * (target_deg_y - self.last_error_y) / dt
        delta_pan = (self.kp * target_deg_x) + d_x
        delta_tilt = (self.kp * target_deg_y) + d_y

        # Angle clipping
        angle_max = 15
        delta_pan = max(-angle_max, min(angle_max, delta_pan))
        delta_tilt = max(-angle_max, min(angle_max, delta_tilt))

        # Update error
        self.last_error_x = target_deg_x
        self.last_error_y = target_deg_y
        self.last_time = current_time

        return delta_pan, delta_tilt

    def perform_overwatch(self):
        """ Simulates the sweeping motion. """
        sweep_speed = 4.0
        sweep_range = 20.0

        if self.overwatch_dir == "left":
            self.current_pan += sweep_speed
            if self.current_pan >= sweep_range:
                self.current_pan = sweep_range
                self.overwatch_dir = "right"
        else:
            self.current_pan -= sweep_speed
            if self.current_pan <= -sweep_range:
                self.current_pan = -sweep_range
                self.overwatch_dir = "left"

        print(f"[SIM-OVERWATCH] Sweeping... Pan: {self.current_pan:+.2f}")

    def update_turret(self, pan_ref, tilt_ref, pixel_err_x, pixel_err_y, dist_cm, fire_cmd):
        """ Models the physics of the turret and logs the telemetry. """
        
        # 1. Lock-in with no target
        if pixel_err_x == 0 and pixel_err_y == 0:
            print(f"[SIM-TRACKING] Locked, no target. Pos: {self.current_pan:+.2f}, {self.current_tilt:+.2f}")
            if fire_cmd != self.is_firing_latched:
                self.is_firing_latched = fire_cmd
            return

        # 2. Distance clipping
        safe_dist = max(dist_cm, 10.0) 

        # 3. Parallax calculation
        parallax_tilt = math.degrees(math.atan2(self.laser_offset_y_cm, safe_dist))

        # 4. Error calculation
        deg_error_x = (pixel_err_x * self.deg_per_pixel_x)
        deg_error_x = deg_error_x + 3 # laser lens error 3 degrees

        deg_error_y = (pixel_err_y * self.deg_per_pixel_y) - parallax_tilt

        # 5. Calculate next steps pan and tilt
        delta_pan, delta_tilt = self.calculate_effort(deg_error_x, deg_error_y)
        self.current_pan = max(-60.0, min(60.0, self.current_pan + delta_pan))
        self.current_tilt = max(-40.0, min(60.0, self.current_tilt + delta_tilt))

        if fire_cmd != self.is_firing_latched:
            self.is_firing_latched = fire_cmd

        # 6. Print Console Telemetry
        print(f"[SIM-TRACKING] Dist={dist_cm:.1f}cm | "
              f"Err(deg): x={deg_error_x:+.2f}, y={deg_error_y:+.2f} | "
              f"Virtual Pos: pan={self.current_pan:+.2f}, tilt={self.current_tilt:+.2f} | " 
              f"Fire: {fire_cmd}")


######################################################################################################################################################################
#                                                                       Threading
######################################################################################################################################################################

class ControllerThread(threading.Thread):
    """
    Autonomous hardware state-machine. 
    Maintains cycle times and manages the communication between the software and the turret.
    """
    def __init__(self, plc):
        super().__init__(name="HardwareThread", daemon=True)
        self.plc = plc
        self.running = True
        self.last_heartbeat_time = time.time()
        self.heartbeat_interval = 5.0  # Print state exactly once per 5 seconds
        
        # State Machine Modes: "OVERWATCH", "TRACKING", "STANDBY"
        self.mode = "OVERWATCH"
        
        # The Mailbox Message System
        self.target_pan = 0.0
        self.target_tilt = 0.0
        self.is_firing = False
        self.update_laser = False
        
        # Autonomous Overwatch State
        self.sweep_pan = 0.0
        self.sweep_tilt = 0.0
        self.sweep_dir = "left"
        self.sweep_speed = 0.0
        self.sweep_range = 20.0

    def set_mode(self, new_mode):
        """ Thread-safe trigger to change turret behavior """
        if self.mode != new_mode:
            log(f"Shifting to {new_mode} mode", "DEBUG")
            self.mode = new_mode

    def update_tracking_target(self, pan, tilt, fire_cmd, update_laser):
        """ The Mailbox Drop: AI overwrites this with the freshest data """
        self.target_pan = pan
        self.target_tilt = tilt
        self.is_firing = fire_cmd
        self.update_laser = update_laser

    def run(self):
        log("Hardware Thread: Active and Autonomous", "INFO")

        # ------------------------------------------ Main Loop ---------------------------------------------------------------------
        try:
            while self.running:
                if not self.plc or not self.plc.connected:
                    time.sleep(0.1)
                    continue

                # --- STATE: OVERWATCH ---
                if self.mode == "OVERWATCH":
                    log("SWITCH: Start overwatch", "INFO")
                    ok, _ = self.plc.send_pose(0, 0)

                    # 1. Sweep logic
                    if self.sweep_dir == "left":
                        self.sweep_pan += self.sweep_speed
                        if self.sweep_pan >= self.sweep_range: self.sweep_dir = "right"
                    else:
                        self.sweep_pan -= self.sweep_speed
                        if self.sweep_pan <= -self.sweep_range: self.sweep_dir = "left"

                    # 2. Wait-and-Verify
                    ok, _ = self.plc.send_pose(pan=float(self.sweep_pan), tilt=float(self.sweep_tilt))

                    time.sleep(0.4)

                # --- STATE: TRACKING ---
                elif self.mode == "TRACKING":
                    # 1. Capture the telemetry dictionary instead of discarding it
                    ok, telemetry = self.plc.send_pose(pan=float(self.target_pan), tilt=float(self.target_tilt))
                    
                    # 2. SEQUENCE CHECK
                    if ok and self.is_firing:
                        self.plc.set_laser(True)
                    elif ok and not self.is_firing:
                        self.plc.set_laser(False)
                        
                    # 3. DIAGNOSTIC HEARTBEAT
                    current_time = time.time()
                    if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                        # Check if the packet actually contains encoder data or a BUSY state
                        if ok and "pan" in telemetry:
    
                            # Extract the function header byte from the hex string
                            packet_header = telemetry["raw_bytes"][:2]
                            
                            # Condition 1: Direct Match for Pose telemetry
                            if packet_header == "01":
                                real_pan = telemetry["pan"]
                                real_tilt = telemetry["tilt"]
                                print(f"[TURRET HEARTBEAT]")
                                print(f"  -> PAN  | Target Command: {self.target_pan:+.2f} deg | Actual Encoder: {real_pan:+.2f} deg")
                                print(f"  -> TILT | Target Command: {self.target_tilt:+.2f} deg | Actual Encoder: {real_tilt:+.2f} deg")
                                
                            # Condition 2: Intercept the Laser confirmation frames
                            elif packet_header == "0c":
                                print(f"[TURRET HEARTBEAT] Warning: PLC is busy/moving (Laser State Sync Overlap)")
                                
                            # Condition 3: Catch any other standard busy signals
                            elif telemetry.get("status") == "BUSY":
                                print(f"[TURRET HEARTBEAT] Warning: PLC is busy/moving")
                            
                        self.last_heartbeat_time = current_time
                    time.sleep(0.03)

        except Exception as e:
            log(f"Hardware Loop Error: {e}", "ERROR")

        finally:
            log("Hardware Thread Dying: Forcing reset...", "WARNING")
            try:
                if self.plc and self.plc.connected:
                    self.plc.set_velocity(tilt_vel=500, pan_vel=500)
                    self.plc.set_laser(False)
                    self.plc.send_pose(pan=0, tilt=0)
                    self.plc.disconnect()
            except: pass

    def stop(self):
        self.running = False

######################################################################################################################################################################
#                                                                       Controller
######################################################################################################################################################################

class RealTurretController:
    """
    Standalone Hardware-active class. 
    Acts purely as a mathematical PD calculator and Mailbox updater.
    """
    def __init__(self, plc_ref):
        # System variables
        self.plc = plc_ref
        self.is_firing_latched = False
        self.target_csv_counter = 0

        # PD gains
        self.pan_kp = 0.06
        self.pan_kd = 0.02

        self.tilt_kp = 0.06
        self.tilt_kd = 0.02

        # Deadzone 
        self.deadzone_deg = 0.5
        self.last_error_x = 0.0
        self.last_error_y = 0.0

        self.last_time = time.time()

        self.current_pan = 0.0
        self.current_tilt = 0.0

        # Pixel-based error 
        self.cam_res_x = config.FRAME_WIDTH
        self.cam_res_y = config.FRAME_HEIGHT
        self.deg_per_pixel_x = 60.0 / self.cam_res_x
        self.deg_per_pixel_y = 40.0 / self.cam_res_y

        # Parallax constants
        self.laser_offset_y_cm = 4.4 # camera-laser vertical difference
        self.laser_offset_x_cm = -5.5 # laser center and lens difference (experimental)

        # Thread
        self.hw_thread = ControllerThread(self.plc)
        self.hw_thread.start()
        log("Controller: PHYSICAL hardware linked via Autonomous Background Thread", "INFO")

    def perform_overwatch(self):
        self.hw_thread.set_mode("OVERWATCH")

    def calculate_effort(self, target_deg_x, target_deg_y):
        """ Calculates how much to turn for this iteration """
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Deadzone calculation
        if dt <= 0.001: dt = 0.033
        if abs(target_deg_x) < self.deadzone_deg: target_deg_x = 0.0
        if abs(target_deg_y) < self.deadzone_deg: target_deg_y = 0.0

        # Position calculation
        d_x = self.pan_kd * (target_deg_x - self.last_error_x) / dt
        d_y = self.tilt_kd * (target_deg_y - self.last_error_y) / dt
        delta_pan = (self.pan_kp * target_deg_x) + d_x
        delta_tilt = (self.tilt_kp * target_deg_y) + d_y

        # Angle clipping
        angle_max = 20
        delta_pan = max(-angle_max, min(angle_max, delta_pan))
        delta_tilt = max(-angle_max, min(angle_max, delta_tilt))

        # Update error
        self.last_error_x = target_deg_x
        self.last_error_y = target_deg_y
        self.last_time = current_time

        return delta_pan, delta_tilt

    def update_turret(self, pan_ref, tilt_ref, pixel_err_x, pixel_err_y, dist_cm, fire_cmd):
        """ Calculates tracking math and drops the result into the Mailbox """        

        # 1. Mode switch
        if self.hw_thread.mode != "TRACKING":
            log("Controller: TRACKING mode active", "DEBUG")
            self.current_pan = self.hw_thread.sweep_pan
            self.current_tilt = self.hw_thread.sweep_tilt
            self.last_error_x = 0.0
            self.last_error_y = 0.0
            self.last_time = time.time()
            self.hw_thread.set_mode("TRACKING")

        # 2. Lock-in with no target, stand still
        if pixel_err_x == 0 and pixel_err_y == 0:
            # Simply update the mailbox with current position to keep laser/fire state
            self.hw_thread.update_tracking_target(
                pan=float(self.current_pan),
                tilt=float(self.current_tilt),
                fire_cmd=fire_cmd,
                update_laser=(fire_cmd != self.is_firing_latched) # Relay state change only
            )
            return

        ## 3. Parallax & PD (Only reached if there IS an error)
        # 3A: Distance clipping
        safe_dist = max(dist_cm, 10.0) # either true distance or 10 cm away

        # 3B: Parallax calculation
        parallax_tilt = math.degrees(math.atan2(self.laser_offset_y_cm, safe_dist))
        #parallax_pan = math.degrees(math.atan2(self.laser_offset_x_cm, safe_dist))

        # 3C: Error calculation
        deg_error_x = (pixel_err_x * self.deg_per_pixel_x)
        deg_error_x = deg_error_x + 3 # laser lens error 3 degrees

        deg_error_y = (pixel_err_y * self.deg_per_pixel_y) - parallax_tilt

        # 3D: Calculate next steps pan and tilt
        delta_pan, delta_tilt = self.calculate_effort(deg_error_x, deg_error_y)
        self.current_pan = max(-60.0, min(60.0, self.current_pan + delta_pan))
        self.current_tilt = max(-40.0, min(60.0, self.current_tilt + delta_tilt))

        # 4. Laser State Management
        update_relay = False
        if fire_cmd != self.is_firing_latched:
            update_relay = True
            self.is_firing_latched = fire_cmd

        # 5. Mailbox Drop
        self.hw_thread.update_tracking_target(
            pan=float(self.current_pan),
            tilt=float(self.current_tilt),
            fire_cmd=fire_cmd,
            update_laser=update_relay
        )

        # Telemetry Logging
        """log_dir = "logs"
        log_file = os.path.join(log_dir, "target_telemetry.csv")
        
        # Ensure the logs directory exists in your workspace
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Check if file exists to determine if we need to write headers
        file_is_empty = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        
        # Append the coordinate data safely
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            target_id = self.target_csv_counter
            self.target_csv_counter += 1
            
            # Write headers only on the very first creation
            if file_is_empty:
                writer.writerow(["id", "pan_ref", "tilt_ref", "camera_x", "camera_y", "pan_pixel_error", "tilt_pixel_error", "pan_deg_error", "tilt_deg_error"])
                
            writer.writerow([target_id, pan_ref, tilt_ref, 960, 540, pixel_err_x, -pixel_err_y, deg_error_x, deg_error_y])"""