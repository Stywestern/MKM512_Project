# modules/interface.py

##################################### Imports #####################################

# Standart Libraries
import sys

# Third Party Libraries
from PyQt6.QtWidgets import (QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2

# Modules
import config
from utils import opencv_to_qpixmap
from modules.utils import log

###################################################################################


class SentryHUD(QMainWindow):
    def __init__(self, worker_ref):
        super().__init__()
        self.worker = worker_ref 
        self.setWindowTitle("Sentry Command Center")
        self.init_ui()
        self.setup_connections() # map UI buttons to logic handlers

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
        self.stop_btn = QPushButton("STOP") # Switch between STOP and RESUME
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
        self.layout.setColumnStretch(0, 4)
        self.layout.setColumnStretch(1, 6)

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
        """ Stops the AI part, enabling us to see the camera unaltered (mostly for fps comparison) """
        is_now_frozen = self.worker.toggle_freeze() # Worker logic

        if is_now_frozen:
            self.stop_btn.setText("RESUME")
            self.stop_btn.setStyleSheet("background-color: #444; color: yellow;")
            self.history_list.append("<font color='yellow'>[SYSTEM PAUSED]</font>")
        else:
            self.stop_btn.setText("STOP")
            self.stop_btn.setStyleSheet("background-color: #222; color: white;")
            self.history_list.append("<font color='white'>[SYSTEM RESUMED]</font>")

    def handle_restart(self):
        """ Purge AI memory, forcing it to run recognition again on all faces """
        #self.history_list.clear()

        self.worker.reset_tracking_data() # Worker logic
        self.history_list.append("<b style='color:cyan;'>[SYSTEM] REBOOT SUCCESSFUL: MEMORY PURGED</b>")
    
    def handle_lock_toggle(self):
        """ Switch between Overwatch and Active Tracking """
        is_locked = self.worker.toggle_lock() # Worker logic
        
        if is_locked:
            self.release_btn.setText("RELEASE")
            self.history_list.append("<b style='color:orange;'>TURRET: LOCK-IN ACQUIRED</b>")
        else:
            self.release_btn.setText("LOCK-IN")
            self.history_list.append("<i style='color:gray;'>TURRET: OVERWATCH MODE</i>")

    def handle_next_target(self):
        """ Switch the current 'Enemy' to the next person in view """
        new_id = self.worker.switch_target(step=1) # Worker logic

        if new_id is not None:
            self.history_list.append(f"Target Switched: Now tracking ID {new_id}")
        else:
            self.history_list.append("<i style='color:gray;'>[WARN] No targets available to cycle</i>")

    def handle_prev_target(self):
        """ Switch the current 'Enemy' to the previous person in view """
        new_id = self.worker.switch_target(step=-1) # Worker logic

        if new_id is not None:
            self.history_list.append(f"Target Switched: Now tracking ID {new_id}")
        else:
            self.history_list.append("<i style='color:gray;'>[WARN] No targets available to cycle</i>")

    def handle_fire(self):
        """ Simulate engagement """
        is_fire = self.worker.trigger_fire() # Worker logic

        if is_fire:
            self.history_list.append("<b style='color:red;'>[ACTION ACCEPTED] WEAPON SYSTEM: FIRE</b>")
        else:
            self.history_list.append("<b style='color:red;'>[ACTION REJECTED] WEAPON SYSTEMS OFFLINE </b>")
    
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
        The main function, changes the screen depending on the incoming data
        main_frame : CameraStream frame with cv2 drawings
        image_package : two np.arrays representing the crop and alignment
        data_package: a dictionary and a float, representing events and fps
        """ 

        # 0. Extract data
        detection_crop, retina_align = image_package[0], image_package[1]
        logs, fps_val = data_package[0], data_package[1]

        # 1. Update the Live Main Feed
        cv2.putText(main_frame, f"FPS: {fps_val}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        self.video_label.setPixmap(opencv_to_qpixmap(main_frame, self.video_label.width(), self.video_label.height())) 

        # 2. Event Parsing
        for event in logs:
            # 1. Print the header (e.g., [IDENTITY] ID 5: Kerem)
            self.history_list.append(event.get("html", ""))
            
            # 2. If recognition, sort and print the scores
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

        # 3. If new detection, update the top left images
        if detection_crop.size > 0 and retina_align.size > 0:

            # A. Update Detection Box
            self.detect_cap.setPixmap(opencv_to_qpixmap(detection_crop, 112, 112))

            # B. Update Alignment Box
            self.align_cap.setPixmap(opencv_to_qpixmap(retina_align, 112, 112))

            # C. Update Comparison Box (the closes embedding model decided on)
            ref_path = None
            for event in reversed(logs):
                if event["type"] == "RECOGNITION":
                    ref_path = event["metadata"].get("ref_path")
                    break
            
            ref_cv = cv2.imread(ref_path)
            if ref_cv is not None:
                self.compare_cap.setPixmap(opencv_to_qpixmap(ref_cv, 112, 112))