# modules/controller.py

##################################### Imports #####################################
# Libraries
import numpy as np
import time
import threading
import queue
import math
import os
import csv

# Modules
from modules.utils import log
import config

###################################################################################

######################################################################################################################################################################
#                                                            SIM CONTROLLER (Digital Twin)
######################################################################################################################################################################

class SimTurretController:
    """
    Standalone Virtual Turret.
    Calculates physical offsets and simulates motor accumulation in the console.
    """
    def __init__(self):
        # 1. PD Parameters
        self.kp = 1.0  
        self.kd = 0.05  
        self.deadzone_deg = 0.5 
        
        # 2. Mathematical Memory
        self.last_error_x = 0
        self.last_error_y = 0
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

    def calculate_effort(self, target_deg_x, target_deg_y):
        """ Standard PD logic operating in degrees. """
        d_x = self.kd * (target_deg_x - self.last_error_x) 
        d_y = self.kd * (target_deg_y - self.last_error_y) 

        delta_pan = (self.kp * target_deg_x) + d_x
        delta_tilt = (self.kp * target_deg_y) + d_y

        self.last_error_x = target_deg_x
        self.last_error_y = target_deg_y
        
        return delta_pan, delta_tilt

    def perform_overwatch(self):
        """ Simulates the sweeping motion. """
        sweep_speed = 2.0
        sweep_range = 10.0

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

        # Just print the virtual state, no network calls
        print(f"[SIM-OVERWATCH] Sweeping... Pan: {self.current_pan:+.2f}")

    def update_turret(self, pixel_err_x, pixel_err_y, dist_cm, fire_cmd):
        """ Models the physics of the turret and prints the telemetry. """
        # 1. Absolute Scaling
        deg_error_x = pixel_err_x * self.deg_per_pixel_x
        deg_error_y = pixel_err_y * self.deg_per_pixel_y

        # 2. Get the delta step
        delta_pan, delta_tilt = self.calculate_effort(deg_error_x, deg_error_y)

        # 3. Deadzone filter
        delta_pan = 0 if abs(delta_pan) < self.deadzone_deg else delta_pan
        delta_tilt = 0 if abs(delta_tilt) < self.deadzone_deg else delta_tilt

        # 4. Accumulate Virtual State
        self.current_pan += delta_pan
        self.current_tilt += delta_tilt

        # 5. Virtual Hardware Clamps
        self.current_pan = max(-60.0, min(60.0, self.current_pan))
        self.current_tilt = max(-40.0, min(40.0, self.current_tilt))

        # 6. Parallax Correction
        parallax_angle = 1.0 
        final_tilt = self.current_tilt - parallax_angle

        # 7. Print Console Telemetry
        """print(f"[SIM-TRACKING] Err(px): x={pixel_err_x:+.1f}, y={pixel_err_y:+.1f} | "
              f"Err(deg): x={deg_error_x:+.2f}, y={deg_error_y:+.2f} | "
              f"Step: pan={delta_pan:+.2f}, tilt={delta_tilt:+.2f} | "
              f"Virtual Pos: pan={self.current_pan:+.2f}, tilt={self.current_tilt:+.2f} | " 
              f"Fire: {fire_cmd}")"""


######################################################################################################################################################################
#                                                                       Threading
######################################################################################################################################################################

class ControllerThread(threading.Thread):
    """
    Autonomous hardware state-machine. 
    Maintains strict cycle times and owns the true physical state of the turret.
    """
    def __init__(self, plc):
        super().__init__(name="HardwareThread", daemon=True)
        self.plc = plc
        self.running = True
        
        # 1. State Machine Modes: "OVERWATCH", "TRACKING", "STANDBY"
        self.mode = "OVERWATCH"
        
        # 2. The Mailbox (Replaces queue.Queue)
        self.target_pan = 0.0
        self.target_tilt = 0.0
        self.is_firing = False
        self.update_laser = False
        
        # 3. Autonomous Overwatch State
        self.sweep_pan = 0.0
        self.sweep_tilt = 0.0
        self.sweep_dir = "left"
        self.sweep_speed = 4.0
        self.sweep_range = 20.0

    def set_mode(self, new_mode):
        """ Thread-safe trigger to change turret behavior """
        if self.mode != new_mode:
            log(f"Hardware Thread: Shifting to {new_mode} mode", "DEBUG")
            self.mode = new_mode

    def update_tracking_target(self, pan, tilt, fire_cmd, update_laser):
        """ The Mailbox Drop: AI overwrites this with the freshest data """
        self.target_pan = pan
        self.target_tilt = tilt
        self.is_firing = fire_cmd
        self.update_laser = update_laser

    def run(self):
        log("Hardware Thread: Active and Autonomous", "INFO")

        try:
            while self.running:
                if not self.plc or not self.plc.connected:
                    time.sleep(0.1)
                    continue

                # --- STATE: OVERWATCH ---
                if self.mode == "OVERWATCH":
                    # 1. Math logic
                    if self.sweep_dir == "left":
                        self.sweep_pan += self.sweep_speed
                        if self.sweep_pan >= self.sweep_range: self.sweep_dir = "right"
                    else:
                        self.sweep_pan -= self.sweep_speed
                        if self.sweep_pan <= -self.sweep_range: self.sweep_dir = "left"

                    # 2. Wait-and-Verify with Backpressure
                    ok, _ = self.plc.send_pose(pan=float(self.sweep_pan), tilt=float(self.sweep_tilt))

                    time.sleep(0.4)

                # --- STATE: TRACKING ---
                elif self.mode == "TRACKING":
                    # 1. ALWAYS send the pose
                    ok, _ = self.plc.send_pose(pan=float(self.target_pan), tilt=float(self.target_tilt))
                    
                    # 2. SEQUENCE CHECK:
                    # Only attempt to fire IF the pose was successful. 
                    # If pose failed (PLC busy/moving), we skip firing this cycle 
                    # so we don't clog the buffer.
                    if ok and self.is_firing:
                        # We send the fire command immediately after a successful pose
                        self.plc.set_laser(True)
                    elif ok and not self.is_firing:
                        # If we aren't firing, ensure laser is explicitly off
                        self.plc.set_laser(False)
                    
                    # If !ok, we do nothing and let the loop restart. 
                    # The Pose command will retry next time.
                    time.sleep(0.03)
                
                else:
                    time.sleep(0.1)

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
        self.plc = plc_ref
        self.is_firing_latched = False
        self.target_csv_counter = 0

        # Unified PD gains
        self.kp = 0.06
        self.kd = 0.02

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

        self.hw_thread = ControllerThread(self.plc)
        self.hw_thread.start()
        log("Controller: PHYSICAL hardware linked via Autonomous Background Thread", "INFO")

    def perform_overwatch(self):
        self.hw_thread.set_mode("OVERWATCH")

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

    def update_turret(self, pan_ref, tilt_ref, pixel_err_x, pixel_err_y, dist_cm, fire_cmd):

        """ Calculates tracking math and drops the result into the Mailbox """        
        # 1. Mode switch
        if self.hw_thread.mode != "TRACKING":
            log("Controller: Executing seamless handoff to TRACKING", "DEBUG")
            self.current_pan = self.hw_thread.sweep_pan
            self.current_tilt = self.hw_thread.sweep_tilt
            self.last_error_x = 0.0
            self.last_error_y = 0.0
            self.last_time = time.time()
            self.hw_thread.set_mode("TRACKING")

        # 2. Lock-in with no target
        if pixel_err_x == 0 and pixel_err_y == 0:
            # Simply update the mailbox with current position to keep laser/fire state
            self.hw_thread.update_tracking_target(
                pan=float(self.current_pan),
                tilt=float(self.current_tilt),
                fire_cmd=fire_cmd,
                update_laser=(fire_cmd != self.is_firing_latched) # Relay state change only
            )
            return

        ## 3. Parallax & PID (Only reached if there IS an error)
        # 3A: Distance clipping
        safe_dist = max(dist_cm, 10.0) # either true distance or 10 cm away

        # 3B: Parallax calculation
        parallax_tilt = math.degrees(math.atan2(self.laser_offset_y_cm, safe_dist))
        #parallax_pan = math.degrees(math.atan2(self.laser_offset_x_cm, safe_dist))
        print("Distance(cm):", dist_cm)
        #print("Laser tilt(degrees):", parallax_pan)

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
        log_dir = "logs"
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
                
            writer.writerow([target_id, pan_ref, tilt_ref, 960, 540, pixel_err_x, -pixel_err_y, deg_error_x, deg_error_y])