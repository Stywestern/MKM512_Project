# modules/detector.py

##################################### Imports #####################################

# Standart Libraries
from datetime import datetime

# Third Party Libraries
import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

# Modules

###################################################################################

# Custom Logger
def log(message, level="INFO"):
    """
    Better print for the project. Adds timestamps and eye-catcher stuff.
    Levels: INFO for standart stuff, WARNING for weird occasions, ERROR for unwanted behaviour.
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")


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
def opencv_to_qpixmap(frame, width, height):
    """
    Utility to convert CV2 BGR images to QPixmap.
    """
    if frame is None or frame.size == 0:
        return QPixmap()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    
    qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    
    return QPixmap.fromImage(qt_img).scaled(
        width, height, 
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )