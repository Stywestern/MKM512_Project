# modules/detector.py

##################################### Imports #####################################
# Libraries
from datetime import datetime

###################################################################################

# Custom Logger
def log(message, level="INFO"):
    """
    Standardized logger for the project.
    Levels: INFO for standart stuff, WARNING for weird occasions, ERROR for unwanted behaviour
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")

# Event logger for Visionworker emitions
def create_event(event_type, **kwargs):
    """
    Standardizes event packaging for the Sentry system.
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