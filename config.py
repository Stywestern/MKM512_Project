# config.py

# --- SYSTEM SETTINGS ---
DEBUG_MODE = True           # Show video window for debugging

# --- CAMERA SETTINGS ---
CAMERA_INDEX = 0           # USB Webcam index for pixels
FRAME_WIDTH = 1280          # Logitech C270 specs
FRAME_HEIGHT = 720
FPS = 30                    # Target framerate

# --- DETECTOR SETTINGS ---
RUN_ON_GPU = True           # Toggle GPU usage
DET_CONF_THRESHOLD = 0.6       # How sure the detector machine should be
REG_CONF_THRESHOLD = 0.5       # How sure the recognizer machine should be, but reversed and between 0-2
RETRY_INTERVAL = 10.0        # Seconds to wait before re-identifying an Unknown