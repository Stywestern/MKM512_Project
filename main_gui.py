#main.py
"""
try:
    import onnxruntime as ort
    # This "pre-warms" the DLL bindings before PyQt6 can interfere
    _ = ort.get_device() 
    print(_)

except Exception as e:
    print(f"Pre-import warning: {e}")
"""

##################################### Imports #####################################
# Libraries
import os
import sys

from PyQt6.QtWidgets import QApplication

# Modules
from modules.camera import CameraStream
from modules.interface import SentryHUD
from modules.visionworker import VisionWorker
from modules.utils import log

###################################################################################

def main():
    log("Initializing QApplication... ", "INFO")
    app = QApplication(sys.argv) # creates the live program, I can have one such application rolling
    log("App initialized", "INFO")

    log("Initializing CameraStream... ", "INFO")
    shared_cam = CameraStream().start() # camera takes time to load, so it should start with the UI components as well
    
    # 1. Create instances
    log("Initializing VisionWorker... ", "INFO")
    worker = VisionWorker(shared_cam) # since it inherits from Qthread it creates a secondary execution context

    log("Initializing SentryHUD... ", "INFO")
    ui = SentryHUD() # builds but not yet draws
    
    # 2. Connect Signals (Worker -> UI)
    log("Initializing worker publisher... ", "INFO")
    worker.update_signal.connect(ui.update_main_feed)
    
    # 3. Connect UI Actions (UI -> Worker)
    log("Binding UI commands... ", "INFO")
    ui.freeze_btn.clicked.connect(lambda: setattr(worker, 'is_frozen', not worker.is_frozen))
    
    # 4. Start
    log("Starting UI... ", "INFO")
    ui.showFullScreen()

    log("Starting Sentry Subsystem...", "INFO")
    worker.start()
    
    sys.exit(app.exec()) # waits until GUI window is closed

if __name__ == "__main__":
    main()