# modules/controller.py

##################################### Imports #####################################
# Libraries
from pycomm3 import LogixDriver
import numpy as np
import time
import threading
import queue
import math

# Modules
from modules.utils import log
import config

###################################################################################

# modules/controller.py

##################################### Imports #####################################
# Libraries
import numpy as np
import time
import threading
import queue
import math

# Modules
from modules.utils import log

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
        print(f"[SIM-TRACKING] Err(px): x={pixel_err_x:+.1f}, y={pixel_err_y:+.1f} | "
              f"Err(deg): x={deg_error_x:+.2f}, y={deg_error_y:+.2f} | "
              f"Step: pan={delta_pan:+.2f}, tilt={delta_tilt:+.2f} | "
              f"Virtual Pos: pan={self.current_pan:+.2f}, tilt={self.current_tilt:+.2f} | " 
              f"Fire: {fire_cmd}")


######################################################################################################################################################################
#                                                                       Threading
######################################################################################################################################################################
class ControllerThread(threading.Thread):
    def __init__(self, plc):
        super().__init__()
        self.plc = plc
        self.command_queue = queue.Queue(maxsize=1) 
        self.running = True
        self.daemon = True 

    def run(self):
        log("Hardware Thread: Active", "INFO")
        
        # 1. The Master TRY Block
        try:
            while self.running:
                try:
                    cmd = self.command_queue.get(timeout=0.05)
                    
                    if cmd['type'] == 'pose':
                        self.plc.send_pose(cmd['pan'], cmd['tilt'])
                        
                    elif cmd['type'] == 'laser':
                        self.plc.set_laser(cmd['state'])
                        
                    elif cmd['type'] == 'turret_state':
                        self.plc.send_pose(cmd['pan'], cmd['tilt'])
                        
                        if cmd['update_laser']:
                            self.plc.set_laser(cmd['laser_state'])
                            
                    elif cmd['type'] == 'dynamics':
                        self.plc.set_velocity(tilt_vel=cmd['vel'], pan_vel=cmd['vel'])
                        self.plc.set_acceleration(tilt_acc=cmd['acc'], pan_acc=cmd['acc'])
                    
                    self.command_queue.task_done()
                    time.sleep(0.01) 

                except queue.Empty:
                    # Optional: Print a dot every few seconds to prove the thread is still looping
                    continue
                except Exception as e:
                    log(f"Queue Error: {e}", "ERROR")
            
        # 2. Finally Block (Guaranteed Hardware Shutdown)
        finally:
            log("Hardware Thread Dying: Forcing reset to 0,0 at Speed 500", "WARNING")
            try:
                self.plc.set_velocity(tilt_vel=500, pan_vel=500)
                self.plc.send_pose(pan=0, tilt=0)
                self.plc.set_laser(False)
                self.plc.disconnect()
                log("Hardware safely disconnected.", "SUCCESS")
            except Exception as cleanup_error:
                log(f"Could not complete safe exit: {cleanup_error}", "ERROR")

    def issue_command(self, cmd):
        """ Non-blocking queue entry. """
        try:
            if self.command_queue.full():
                self.command_queue.get_nowait()
            
            self.command_queue.put_nowait(cmd)
            
        except queue.Full:
            pass

    def clear_queue(self):
        """ Instantly drops all pending hardware commands """
        with self.command_queue.mutex:
            self.command_queue.queue.clear()
        log("Hardware Thread: Command queue purged.", "WARNING")

    def stop(self):
        """ Signals the thread to finish """
        self.running = False



######################################################################################################################################################################
#                                                           REAL CONTROLLER (Hardware) 
######################################################################################################################################################################

class RealTurretController:
    """
    Standalone Hardware-active class. 
    Handles all PD math, kinematic state, and hardware thread queuing internally.
    """
    def __init__(self, plc_ref):
        self.plc = plc_ref
        self.is_firing_latched = False
        
        # 1. PD Parameters
        self.kp = 0.06      
        self.kd = 0.02      
        self.deadzone_deg = 0.5 
        
        # 2. Mathematical Memory
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_time = time.time()
        
        # 3. Unified Kinematic State
        self.current_pan = 0.0  
        self.current_tilt = 0.0
        self.overwatch_dir = "left"

        # 4. Camera Physical Properties
        self.cam_res_x = config.FRAME_WIDTH
        self.cam_res_y = config.FRAME_HEIGHT
        self.fov_pan = 60.0   
        self.fov_tilt = 40.0  

        self.deg_per_pixel_x = self.fov_pan / self.cam_res_x
        self.deg_per_pixel_y = self.fov_tilt / self.cam_res_y
        
        # 5. PHYSICAL HARDWARE OFFSETS (in cm)
        # Measure the distance from the center of your camera lens to the center of your laser.
        # Assuming laser is mounted 5cm BELOW the camera, and perfectly centered horizontally.
        self.laser_offset_y_cm = 4.4  
        self.laser_offset_x_cm = -5.5  

        # Start Hardware Thread
        self.hw_thread = ControllerThread(self.plc)
        self.hw_thread.start()

        log("Controller: PHYSICAL hardware linked via Background Thread", "INFO")

    def perform_overwatch(self):
        # [Unchanged from your current file]
        if not self.plc or not self.plc.connected:
            return
            
        if abs(self.current_tilt) > 0.1:
            self.current_tilt = 0.0
            self.hw_thread.issue_command({
                'type': 'pose',
                'pan': float(self.current_pan),
                'tilt': 0.0
            })
            return
        
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

        self.hw_thread.issue_command({
            'type': 'pose', 
            'pan': float(self.current_pan), 
            'tilt': float(self.current_tilt)
        })

    def calculate_effort(self, target_deg_x, target_deg_y):

        # [Unchanged from your current file]
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.001: dt = 0.033

        if abs(target_deg_x) < self.deadzone_deg: target_deg_x = 0.0
        if abs(target_deg_y) < self.deadzone_deg: target_deg_y = 0.0

        d_x = self.kd * (target_deg_x - self.last_error_x) / dt
        d_y = self.kd * (target_deg_y - self.last_error_y) / dt

        delta_pan = (self.kp * target_deg_x) + d_x
        delta_tilt = (self.kp * target_deg_y) + d_y

        angle_max = 10
        delta_pan = max(-angle_max, min(angle_max, delta_pan))
        delta_tilt = max(-angle_max, min(angle_max, delta_tilt))

        self.last_error_x = target_deg_x
        self.last_error_y = target_deg_y
        self.last_time = current_time

        return delta_pan, delta_tilt


    def update_turret(self, pixel_err_x, pixel_err_y, dist_cm, fire_cmd):
        # 1. Prevent math errors if distance drops to zero or goes negative
        safe_dist = max(dist_cm, 1.0)  
        
        # 2. Calculate the exact Parallax angle required
        parallax_tilt = math.degrees(math.atan2(self.laser_offset_y_cm, safe_dist))
        parallax_pan = math.degrees(math.atan2(self.laser_offset_x_cm, safe_dist))

        # 3. Convert raw pixels to degrees
        deg_error_x = pixel_err_x * self.deg_per_pixel_x
        deg_error_y = pixel_err_y * self.deg_per_pixel_y

        if not (pixel_err_x == 0 and pixel_err_y == 0):
            deg_error_x -= parallax_pan
            deg_error_y -= parallax_tilt

        # 4. Calculate PID effort using the shifted error
        delta_pan, delta_tilt = self.calculate_effort(deg_error_x, deg_error_y)

        # 5. Apply directly to physical state. No more end-of-line hacks.
        self.current_pan += delta_pan
        self.current_tilt += delta_tilt

        self.current_pan = max(-60.0, min(60.0, self.current_pan))
        self.current_tilt = max(-40.0, min(40.0, self.current_tilt))

        print(f"[HW-TRACKING] Err(px): x={pixel_err_x:+.1f}, y={pixel_err_y:+.1f} | "
              f"Dist: {dist_cm:.1f}cm | "
              f"Parallax(deg): {parallax_tilt:.2f} | "
              f"Step: pan={delta_pan:+.2f}, tilt={delta_tilt:+.2f} | "
              f"Fire: {fire_cmd}")

        # 6. Send the raw current state directly to the PLC
        if self.plc and self.plc.connected:
            update_relay = False

            if fire_cmd and not self.is_firing_latched:
                update_relay = True
                self.is_firing_latched = True

            elif not fire_cmd and self.is_firing_latched:
                update_relay = True
                self.is_firing_latched = False

            self.hw_thread.issue_command({
                'type': 'turret_state', 
                'pan': float(self.current_pan), 
                'tilt': float(self.current_tilt),
                'update_laser': update_relay,
                'laser_state': fire_cmd
            })

######################################################################################################################################################################
#                                                           KINEMATIC CONTROLLER (Hardware) 
######################################################################################################################################################################

class KinematicTurretController:
    """
    Advanced Kinematic Hardware-active class. 
    Uses true 3D spatial mapping and Inverse Kinematics to calculate angular errors.
    """
    def __init__(self, plc_ref):
        self.plc = plc_ref
        self.is_firing_latched = False

        # 1. PD Parameters
        self.kp = 0.065      
        self.kd = 0.02      
        self.deadzone_deg = 1.5 

        # 2. Mathematical Memory
        self.last_error_pan = 0.0
        self.last_error_tilt = 0.0
        self.last_time = time.time()

        # 3. Unified Kinematic State
        self.current_pan = 0.0  
        self.current_tilt = 0.0
        self.overwatch_dir = "left"

        # 4. Camera & Hardware Physical Properties
        self.focal_length = 1573.0 
        self.laser_offset_y_cm = 4.4  
        self.laser_offset_x_cm = 0.0  

        # --- THE FIX 1: HARDWARE POLARITY ---
        # If an axis runs away from the target, change its multiplier from 1 to -1
        self.pan_polarity = 1  
        self.tilt_polarity = 1 

        # Start Hardware Thread
        self.hw_thread = ControllerThread(self.plc)
        self.hw_thread.start()

        log("Kinematic Controller: PHYSICAL hardware linked via Background Thread", "INFO")

    def calculate_effort(self, raw_pan_err, raw_tilt_err):
        """ Applies PD logic safely to the true spatial angular error. """
        current_time = time.time()
        
        # --- THE FIX 2: PREVENT DERIVATIVE EXPLOSION ---
        # Cap the minimum delta-time at ~30fps to prevent math from blowing up to infinity
        dt = max(current_time - self.last_time, 0.033)

        # 1. Calculate true Derivative BEFORE deadzones corrupt the math
        d_pan = self.kd * (raw_pan_err - self.last_error_pan) / dt
        d_tilt = self.kd * (raw_tilt_err - self.last_error_tilt) / dt

        # 2. Update memory with TRUE error
        self.last_error_pan = raw_pan_err
        self.last_error_tilt = raw_tilt_err
        self.last_time = current_time

        # 3. Apply Deadzone to the Proportional term only
        p_pan_err = 0.0 if abs(raw_pan_err) < self.deadzone_deg else raw_pan_err
        p_tilt_err = 0.0 if abs(raw_tilt_err) < self.deadzone_deg else raw_tilt_err

        # 4. Final PD Summation
        delta_pan = (self.kp * p_pan_err) + d_pan
        delta_tilt = (self.kp * p_tilt_err) + d_tilt

        # 5. Bulletproof Clamps (Ensure max/min are perfectly formatted)
        delta_pan = max(-10.0, min(10.0, delta_pan))
        delta_tilt = max(-10.0, min(10.0, delta_tilt))

        return delta_pan, delta_tilt

    def perform_overwatch(self):
        # [Keep your exact overwatch code here]
        if not self.plc or not self.plc.connected:
            return

        if abs(self.current_tilt) > 0.1:
            self.current_tilt = 0.0
            self.hw_thread.issue_command({'type': 'pose', 'pan': float(self.current_pan), 'tilt': 0.0})
            return

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

        self.hw_thread.issue_command({'type': 'pose', 'pan': float(self.current_pan), 'tilt': float(self.current_tilt)})

    def update_turret(self, pixel_err_x, pixel_err_y, dist_cm, fire_cmd):
        """ Kinematic Execution with Polarity Mapping """
        safe_dist = max(dist_cm, 1.0)

        target_x_cam_cm = (pixel_err_x * safe_dist) / self.focal_length
        target_y_cam_cm = (pixel_err_y * safe_dist) / self.focal_length

        target_x_laser_cm = target_x_cam_cm - self.laser_offset_x_cm
        target_y_laser_cm = target_y_cam_cm + self.laser_offset_y_cm 

        # --- APPLY POLARITY TO THE KINEMATICS ---
        raw_pan_err = math.degrees(math.atan2(target_x_laser_cm, safe_dist)) * self.pan_polarity
        raw_tilt_err = math.degrees(math.atan2(target_y_laser_cm, safe_dist)) * self.tilt_polarity

        delta_pan, delta_tilt = self.calculate_effort(raw_pan_err, raw_tilt_err)

        self.current_pan += delta_pan
        self.current_tilt += delta_tilt

        self.current_pan = max(-60.0, min(60.0, self.current_pan))
        self.current_tilt = max(-40.0, min(40.0, self.current_tilt))

        print(f"[HW-KINEMATIC] Err(px): x={pixel_err_x:+.1f}, y={pixel_err_y:+.1f} | "
              f"Laser Err(deg): pan={raw_pan_err:+.2f}, tilt={raw_tilt_err:+.2f} | "
              f"Step: pan={delta_pan:+.2f}, tilt={delta_tilt:+.2f} | "
              f"Fire: {fire_cmd}")

        if self.plc and self.plc.connected:
            update_relay = False

            if fire_cmd and not self.is_firing_latched:
                update_relay = True
                self.is_firing_latched = True

            elif not fire_cmd and self.is_firing_latched:
                update_relay = True
                self.is_firing_latched = False

            self.hw_thread.issue_command({
                'type': 'turret_state', 
                'pan': float(self.current_pan), 
                'tilt': float(self.current_tilt),
                'update_laser': update_relay,
                'laser_state': fire_cmd
            })