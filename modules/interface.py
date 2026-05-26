# modules/interface.py

##################################### Imports #####################################

# Standart Libraries
import sys
import numpy as np

# Third Party Libraries
from PyQt6.QtWidgets import (QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2

# Modules
from modules.utils import log, opencv_to_qpixmap

###################################################################################

class SentryHUD(QMainWindow):
    def __init__(self, worker_ref):
        super().__init__()
        self.worker = worker_ref 
        self.setWindowTitle("Sentry Command Center")
        self.init_ui()
        self.setup_connections() # map UI buttons to logic handlers

    def closeEvent(self, event):
        """ Catches the window close button (X) to ensure hardware safety """
        
        # If the worker exists, trigger the deterministic shutdown
        if hasattr(self, 'worker'):
            self.worker.shutdown()
            
            # Wait for the QThread to finish cleanly before destroying the window
            self.worker.quit()
            self.worker.wait(2000) 
            
        # Accept the event to allow the window to finally close
        event.accept()

    ###################################################################################
    #                                 LAYOUT
    ###################################################################################
 
    def init_ui(self):
        # Main Container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main Grid Layout
        self.layout = QGridLayout(self.central_widget)

        # --- LEFT COLUMN: DATA & PIPELINE (Width Factor: 4) ---
        self.left_col = QVBoxLayout()

        # A. Pipeline Visuals [DETECT | ALIGN | COMPARE]
        self.pipeline_layout = QHBoxLayout()
        self.detect_cap = self._create_preview_box("DETECT")
        self.align_cap = self._create_preview_box("ALIGN")
        self.compare_cap = self._create_preview_box("COMPARE")
        
        self.pipeline_layout.addWidget(self.detect_cap)
        self.pipeline_layout.addWidget(self.align_cap)
        self.pipeline_layout.addWidget(self.compare_cap)
        self.left_col.addLayout(self.pipeline_layout)

        # B. Detection History
        self.history_label = QLabel("DETECTION HISTORY")
        self.history_label.setStyleSheet("font-weight: bold; color: #00FF00;")
        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        self.history_list.setStyleSheet("background-color: #111; color: #00FF00; font-family: Consolas;")
        
        self.left_col.addWidget(self.history_label)
        self.left_col.addWidget(self.history_list)

        # --- RIGHT COLUMN: CAMERA & CONTROLS (Width Factor: 6) ---
        self.right_col = QVBoxLayout()

        # A. Main Camera (Top 40% of the right side essentially)
        self.video_label = QLabel("INITIALIZING CAMERA...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.video_label.setMinimumSize(640, 480)
        
        # B. Control Buttons (Bottom)
        self.controls_wrapper = QVBoxLayout() # Vertical container for all button rows

            # --- ROW 1: PRIMARY ACTIONS ---
        self.primary_btn_layout = QHBoxLayout()
        self.stop_btn = QPushButton("RESUME") # Switch between STOP and RESUME
        self.restart_btn = QPushButton("RESTART")
        self.release_btn = QPushButton("LOCK-IN") # Switch between LOCK-IN and RELEASE
        self.fire_btn = QPushButton("FIRE")

        # Style the primary buttons
        for btn in [self.stop_btn, self.restart_btn, self.release_btn, self.fire_btn]:
            btn.setMinimumHeight(45)
            self.primary_btn_layout.addWidget(btn)

            # --- ROW 2: TARGET NAVIGATION (Centered) ---
        self.nav_btn_layout = QHBoxLayout()
        self.back_btn = QPushButton("<<")
        self.next_btn = QPushButton(">>")
        
        self.back_btn.setFixedSize(60, 40)
        self.next_btn.setFixedSize(60, 40)
        self.nav_btn_layout.addStretch()
        self.nav_btn_layout.addWidget(self.back_btn)
        self.nav_btn_layout.addWidget(self.next_btn)
        self.nav_btn_layout.addStretch()

        # Combine Rows into the Wrapper
        self.controls_wrapper.addLayout(self.primary_btn_layout)
        self.controls_wrapper.addLayout(self.nav_btn_layout)

        # Add everything to Right Column
        self.right_col.addWidget(self.video_label, 8)    # Increase camera weight to 80%
        self.right_col.addLayout(self.controls_wrapper, 2) # Buttons take 20%

        # --- SET GENERAL LAYOUT ---
        self.layout.addLayout(self.left_col, 0, 0)
        self.layout.addLayout(self.right_col, 0, 1)
        self.layout.setColumnStretch(0, 3) # Changed from 4 to 3
        self.layout.setColumnStretch(1, 7) # Changed from 6 to 7

        log("SentryHUD Layout Anchored", "INFO")

    ###################################################################################
    #                                 BUTTONS
    ###################################################################################

    def setup_connections(self):
        """
        Define buttons and handlers here, some of them will delagate their jobs to the Worker as well.
        Specifically, if a button manages UI elements, its logic stays in here, otherwise it goes to Worker class
        """
        self.stop_btn.clicked.connect(self.handle_stop)
        self.restart_btn.clicked.connect(self.handle_restart)
        self.next_btn.clicked.connect(self.handle_next_target)
        self.back_btn.clicked.connect(self.handle_prev_target)
        self.release_btn.clicked.connect(self.handle_lock_toggle)
        self.fire_btn.clicked.connect(self.handle_fire)

    def handle_stop(self):
        """ Stops the AI part, but prevents pausing if actively firing. """
        
        # SAFETY CHECK: Cannot pause system while weapon is actively firing
        if getattr(self.worker, 'is_firing', False):
            self.history_list.append("<b style='color:red;'>[ERROR] CEASE FIRE BEFORE PAUSING SYSTEM</b>")
            return

        is_now_frozen = self.worker.toggle_freeze() 

        if is_now_frozen:
            self.stop_btn.setText("RESUME") 
            self.history_list.append("[SYSTEM PAUSED]")

            # Lock out dangerous controls while paused
            self.release_btn.setEnabled(False)
            self.fire_btn.setEnabled(False)
        else:
            self.stop_btn.setText("STOP")
            self.history_list.append("[SYSTEM RESUMED]")

            # Restore state based on whether we were locked or not
            self.release_btn.setEnabled(True)
            if self.worker.is_locking:
                self.fire_btn.setEnabled(True)
    
    def handle_restart(self):
        # Prevent restart if actively firing
        if getattr(self.worker, 'is_firing', False):
            self.history_list.append("<b style='color:red;'>[ERROR] CEASE FIRE BEFORE REBOOTING</b>")
            return
            
        self.worker.reset_tracking_data() 
        self.history_list.append("<b style='color:cyan;'>[SYSTEM] REBOOT SUCCESSFUL: MEMORY PURGED</b>")
        
        # Reset UI states to safe defaults
        self.release_btn.setText("LOCK-IN")
        self.fire_btn.setEnabled(False)
        self.fire_btn.setStyleSheet("background-color: #333; color: #777;")
    
    def handle_lock_toggle(self):
        # SAFETY CHECK: Cannot release lock while weapon is firing
        if getattr(self.worker, 'is_firing', False):
            self.history_list.append("<b style='color:red;'>[ERROR] CANNOT DROP LOCK WHILE FIRING!</b>")
            return

        is_locked = self.worker.toggle_lock() 
        
        if is_locked:
            self.release_btn.setText("RELEASE")
            self.history_list.append("<b style='color:orange;'>TURRET: LOCK-IN ACQUIRED</b>")

            # Enable firing now that we have a lock
            self.fire_btn.setEnabled(True)
            self.fire_btn.setStyleSheet("") # Reset to default OS style
            
            self.restart_btn.setEnabled(False)
            self.restart_btn.setStyleSheet("background-color: #333; color: #777;")

            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet("background-color: #333; color: #777;")

        else:
            self.release_btn.setText("LOCK-IN")
            self.history_list.append("<i style='color:gray;'>TURRET: OVERWATCH MODE</i>")

            # Disable firing since lock is dropped
            self.fire_btn.setEnabled(False)
            self.fire_btn.setStyleSheet("background-color: #333; color: #777;")
            
            self.restart_btn.setEnabled(True)
            self.restart_btn.setStyleSheet("")

            self.stop_btn.setEnabled(True)
            self.stop_btn.setStyleSheet("")

    def handle_next_target(self):
        new_id = self.worker.switch_target(step=1) 
        if new_id is not None:
            self.history_list.append(f"Target Switched: Now tracking ID {new_id}")
        else:
            self.history_list.append("<i style='color:gray;'>[WARN] No targets available to cycle</i>")

    def handle_prev_target(self):
        new_id = self.worker.switch_target(step=-1) 
        if new_id is not None:
            self.history_list.append(f"Target Switched: Now tracking ID {new_id}")
        else:
            self.history_list.append("<i style='color:gray;'>[WARN] No targets available to cycle</i>")

    def handle_fire(self):
        is_fire = self.worker.trigger_fire() 

        if is_fire:
            self.history_list.append("<b style='color:red;'>[ACTION ACCEPTED] WEAPON SYSTEM: FIRING</b>")
            self.fire_btn.setText("CEASE FIRE")
            self.fire_btn.setStyleSheet("background-color: darkred; color: white; font-weight: bold;")
            # Disable lock release to force user to cease fire first
            self.release_btn.setEnabled(False)
        else:
            self.history_list.append("<b style='color:green;'>[ACTION ACCEPTED] WEAPONS SAFE</b>")
            self.fire_btn.setText("FIRE")
            self.fire_btn.setStyleSheet("") 
            # Safe to drop lock again
            self.release_btn.setEnabled(True)
    
    ###################################################################################
    #                                 UI UPDATES
    ###################################################################################

    def _create_preview_box(self, text):
        """ Boxes on the top left, for detection comparison """
        lbl = QLabel(text)
        lbl.setFixedSize(112, 112)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        lbl.setStyleSheet("border: 1px solid #555; background-color: #222; color: white; font-size: 10px;")
        return lbl
    
    def update_displays(self, main_frame, image_package, data_package):
        """
        The main function, changes the screen depending on the incoming data.
        Wrapped in a try/except to prevent OpenCV/Array errors from crashing the PyQt window.
        """ 
        try:
            # 0. Extract data
            detection_crop, retina_align = image_package[0], image_package[1]
            logs = data_package[0]
            fps_val = data_package[1]
            
            # Unpack
            system_state = data_package[2] if len(data_package) > 2 else {"has_lock": False, "is_firing": False}

            # 1. Lock-In button only resets if the INTENT to lock is revoked by the operator
            if not system_state.get("is_locking", False) and self.release_btn.text() == "RELEASE":
                self.release_btn.setText("LOCK-IN")
                # Only if the whole system disarms do we reset the fire button visually
                self.fire_btn.setText("FIRE")
                self.fire_btn.setEnabled(False)
                self.fire_btn.setStyleSheet("background-color: #333; color: #777;")
            
            # 2. Dynamic Trigger Safety
            if system_state.get("is_locking", False):
                self.fire_btn.setEnabled(True)
                if not system_state.get("is_firing", False):
                    self.fire_btn.setStyleSheet("")

            # 3. Update the Live Main Feed
            cv2.putText(main_frame, f"FPS: {fps_val}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Convert raw frame to QPixmap without forcing dimensions yet
            raw_pixmap = opencv_to_qpixmap(main_frame) 
            if raw_pixmap:
                scaled_pixmap = raw_pixmap.scaled(
                    self.video_label.width(), 
                    self.video_label.height(), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)

            # 2. Event Parsing
            for event in logs:
                # Print the header (e.g., [IDENTITY] ID 5: Kerem)
                self.history_list.append(event.get("html", ""))
                
                # If recognition, sort and print the scores
                if event["type"] == "RECOGNITION":
                    dists = event["metadata"].get("distances", {})
                    
                    if dists:
                        sorted_candidates = sorted(dists.items(), key=lambda x: x[1])
                        self.history_list.append("<font color='#55FF55'>&nbsp;&nbsp;Ranked Candidates:</font>")
                        
                        for i, (fname, d) in enumerate(sorted_candidates[:20]): # Top 20
                            color = "#FFFFFF" if i == 0 else "#888888"
                            self.history_list.append(
                                f"<font color='{color}' size='2'>&nbsp;&nbsp;&nbsp;&nbsp;{i+1}. {fname}: {d:.4f}</font>"
                            )

            # 3. Side Previews
            if detection_crop.size > 0 and retina_align.size > 0:

                # A. Update Detection Box
                self.detect_cap.setPixmap(opencv_to_qpixmap(detection_crop, 112, 112))

                # B. Update Alignment Box
                self.align_cap.setPixmap(opencv_to_qpixmap(retina_align, 112, 112))

                # C. Update Comparison Box
                ref_path = None
                for event in reversed(logs):
                    if event["type"] == "RECOGNITION":
                        ref_path = event["metadata"].get("ref_path")
                        break
                
                if ref_path and isinstance(ref_path, str):
                    # Nested try/except to prevent IO errors from killing the display update
                    try:
                        ref_cv = cv2.imdecode(
                            np.fromfile(ref_path, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        
                        if ref_cv is not None:
                            self.compare_cap.setPixmap(opencv_to_qpixmap(ref_cv, 112, 112))
                    except Exception as img_err:
                        print(f"Failed to load reference image from disk: {img_err}")

        except Exception as e:
            # Catch all rendering and logic errors so the GUI remains interactive
            print(f"UI Display Loop Exception: {e}")