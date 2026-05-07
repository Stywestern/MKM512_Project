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
                        # 1. Move the motors
                        self.plc.send_pose(cmd['pan'], cmd['tilt'])
                        
                        # 2. Fire the laser (only if the state toggled)
                        if cmd['update_laser']:
                            self.plc.set_laser(cmd['laser_state'])
                            
                    elif cmd['type'] == 'dynamics':
                        # Optional state transitioning hook
                        self.plc.set_velocity(tilt_vel=cmd['vel'], pan_vel=cmd['vel'])
                        self.plc.set_acceleration(tilt_acc=cmd['acc'], pan_acc=cmd['acc'])
                    
                    self.command_queue.task_done()
                    
                    # Smoothing governor
                    time.sleep(0.01) 

                except queue.Empty:
                    continue
                except Exception as e:
                    log(f"Queue Error: {e}", "ERROR")
                    
        except Exception as e:
            # Catches any fatal errors that break the while loop
            log(f"Thread crashed: {e}", "ERROR")
            
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
        self.cam_res_x = 720.0
        self.cam_res_y = 1280.0
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
#                                                           REAL CONTROLLER (Hardware)
######################################################################################################################################################################

class RealTurretController:
    """
    Hardware-active class. 
    Uses a background thread to prevent socket blocking from killing AI FPS.
    """
    def __init__(self, plc_ref):
        self.plc = plc_ref
        self.is_firing_latched = False
        
        # 1. PD Parameters
        self.kp = 1.0  
        self.kd = 0.05  
        self.deadzone_deg = 0.5 
        
        # 2. Mathematical Memory
        self.last_error_x = 0
        self.last_error_y = 0
        
        # 3. Unified Kinematic State
        self.current_pan = 0.0  
        self.current_tilt = 0.0
        self.overwatch_dir = "left"

        # 4. Camera Physical Properties
        self.cam_res_x = 720.0
        self.cam_res_y = 1280.0
        self.fov_pan = 60.0   
        self.fov_tilt = 40.0  

        self.deg_per_pixel_x = self.fov_pan / self.cam_res_x
        self.deg_per_pixel_y = self.fov_tilt / self.cam_res_y

        # Start Hardware Thread
        self.hw_thread = ControllerThread(self.plc)
        self.hw_thread.start()

        log("Controller: PHYSICAL hardware linked via Background Thread", "INFO")

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
        """ Sweeps the unified pan variable and sends to PLC. """
        if not self.plc or not self.plc.connected:
            return
        
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

        self.hw_thread.issue_command({
            'type': 'pose', 
            'pan': float(self.current_pan), 
            'tilt': float(self.current_tilt)
        })

    def update_turret(self, pixel_err_x, pixel_err_y, dist_cm, fire_cmd):
        """ Translates absolute error to degrees and controls hardware. """
        deg_error_x = pixel_err_x * self.deg_per_pixel_x
        deg_error_y = pixel_err_y * self.deg_per_pixel_y

        delta_pan, delta_tilt = self.calculate_effort(deg_error_x, deg_error_y)

        delta_pan = 0 if abs(delta_pan) < self.deadzone_deg else delta_pan
        delta_tilt = 0 if abs(delta_tilt) < self.deadzone_deg else delta_tilt

        self.current_pan += delta_pan
        self.current_tilt += delta_tilt

        self.current_pan = max(-60.0, min(60.0, self.current_pan))
        self.current_tilt = max(-40.0, min(40.0, self.current_tilt))

        parallax_angle = 1.0 
        final_tilt = self.current_tilt - parallax_angle

        print(f"[HW-TRACKING] Err(px): x={pixel_err_x:+.1f}, y={pixel_err_y:+.1f} | "
              f"Err(deg): x={deg_error_x:+.2f}, y={deg_error_y:+.2f} | "
              f"Step: pan={delta_pan:+.2f}, tilt={delta_tilt:+.2f} | "
              f"New Pos: pan={self.current_pan:+.2f}, tilt={self.current_tilt:+.2f} | " 
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
                'tilt': float(final_tilt),
                'update_laser': update_relay,
                'laser_state': fire_cmd
            })