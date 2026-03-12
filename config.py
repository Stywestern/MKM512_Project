# config.py

# --- SYSTEM SETTINGS ---
DEBUG_MODE = True               # Show video window for debugging

# --- CAMERA SETTINGS ---
CAMERA_INDEX = 0                # USB Webcam index for pixels
FRAME_WIDTH = 1280              # Logitech C270 specs
FRAME_HEIGHT = 720
FPS = 30                        # Target framerate
FOCAL_LENGTH = 150 * 65.29 / 6.3   # Focal distance of the cam

# --- DETECTOR SETTINGS ---
RUN_ON_GPU = True               # Toggle GPU usage
DET_CONF_THRESHOLD = 0.25       # How sure the detector machine should be
REG_CONF_THRESHOLD = 0.45       # How sure the recognizer machine should be, but reversed and between 0-2
RETRY_INTERVAL = 10.0           # Seconds to wait before re-identifying an Unknown


# --- DATABASE SETTINGS ---
ENEMIES = [ 'George_W_Bush', 'Gerhard_Schroeder', 'Gloria_Macapagal_Arroyo', 'Hugo_Chavez', 'Hu_Jintao', 'Jennifer_Lopez', 'Kerem_Cantimur', 'Tony_Blair', 'Venus_Williams']
FRIENDS = ['Angelina_Jolie', 'Colin_Powell', 'Kofi_Annan', 'Laura_Bush', 'Megawati_Sukarnoputri', 'Roh_Moo-hyun', 'Serena_Williams', 'Tiger_Woods', 'Vicente_Fox', 'Winona_Ryder'] 

# Tactical Colors (BGR for OpenCV)
COLOR_STRANGER = (255, 165, 0)  # Orange
COLOR_ENEMY = (0, 0, 255)       # Red
COLOR_FRIEND = (0, 255, 0)      # Green