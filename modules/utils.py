# modules/detector.py

##################################### Imports #####################################

# Standart Libraries
from datetime import datetime
import threading
import inspect

# Third Party Libraries
import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

# Modules

###################################################################################

# Custom Logger
def log(message, level="INFO"):
    # 1. Get the name of the thread currently executing this line
    thread_name = threading.current_thread().name
    
    # 2. Inspect the call stack to find out WHO called this log function
    # stack()[1] is the caller. stack()[0] is this log function itself.
    caller_frame = inspect.stack()[1]
    caller_func = caller_frame.function
    caller_file = caller_frame.filename.split('/')[-1].split('\\')[-1] # Gets just the filename
    
    # 3. Format the timestamp
    time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Output example: [14:02:33.105] [INFO] [VisionWorkerThread] <update_turret> (visionworker.py): Calculating PD...
    print(f"[{time_str}] [{level}] [{thread_name}] <{caller_func}> ({caller_file}): {message}")


# Event logger for Visionworker emitions
def create_event(event_type: str, **kwargs):
    """
    Standardizes event packaging for the Sentry system. I use this to send events to the UI.
    Types: 'LOG', 'RECOGNITION', 'LOCK'
    """

    event = {"type": event_type, "metadata": kwargs}
    
    if event_type == "LOG":
        msg = kwargs.get("message", "")
        color = kwargs.get("color", "white")
        event["html"] = f"<font color='{color}'>{msg}</font>"

    elif event_type == "RECOGNITION":
        track_id = kwargs.get("track_id")
        name = kwargs.get("name")
        dists = kwargs.get("distances", {})
        best_dist = 1.0
        
        if dists:
            best_dist = min(dists.values()) if dists else 1.0

        event["html"] = f"<b style='color:cyan;'>[IDENTITY] ID {track_id}: {name} ({best_dist:.2f})</b>"
        
    elif event_type == "LOCK":
        track_id = kwargs.get("track_id")
        status = kwargs.get("status", "LOCKED")
        color = "orange" if status == "LOCKED" else "gray"
        event["html"] = f"<b style='color:{color};'>[SENTRY] {status}: ID {track_id}</b>"

    return event

# Cleanup for interface.py 
def opencv_to_qpixmap(frame, width=None, height=None):
    """
    Utility to convert CV2 BGR images to QPixmap.
    If width and height are provided, safely scales while preserving aspect ratio.
    """
    if frame is None or frame.size == 0:
        return QPixmap()

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    
    # Create the Qt Image (using .copy() prevents memory corruption)
    qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qt_img)
    
    # If target dimensions are provided, scale it cleanly
    if width is not None and height is not None:
        return pixmap.scaled(
            width, height, 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
    # Otherwise, return the unscaled raw pixmap
    return pixmap