# modules/interface.py

##################################### Imports #####################################
# Libraries
from PyQt6.QtWidgets import (QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2

# Modules
import config
from modules.utils import log

###################################################################################

class SentryHUD(QMainWindow):
    def __init__(self, worker_ref):
        super().__init__()
        self.worker = worker_ref 
        self.setWindowTitle("Sentry Command Center")
        self.init_ui()
        self.setup_connections() # method for wiring

    def init_ui(self):
        # Main Container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main Grid Layout
        self.layout = QGridLayout(self.central_widget)

        # --- LEFT COLUMN: DATA & PIPELINE (Width Factor: 4) ---
        self.left_col = QVBoxLayout()

        # 1. Pipeline Visuals [YOLO | ALIGN | COMPARE]
        self.pipeline_layout = QHBoxLayout()
        self.yolo_cap = self._create_preview_box("YOLO")
        self.align_cap = self._create_preview_box("ALIGN")
        self.compare_cap = self._create_preview_box("COMPARE")
        
        self.pipeline_layout.addWidget(self.yolo_cap)
        self.pipeline_layout.addWidget(self.align_cap)
        self.pipeline_layout.addWidget(self.compare_cap)
        self.left_col.addLayout(self.pipeline_layout)

        # 2. Detection History
        self.history_label = QLabel("DETECTION HISTORY")
        self.history_label.setStyleSheet("font-weight: bold; color: #00FF00;")
        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        self.history_list.setStyleSheet("background-color: #111; color: #00FF00; font-family: Consolas;")
        
        self.left_col.addWidget(self.history_label)
        self.left_col.addWidget(self.history_list)

        # --- RIGHT COLUMN: CAMERA & CONTROLS (Width Factor: 6) ---
        self.right_col = QVBoxLayout()

        # 3. Main Camera (Top 40% of the right side essentially)
        self.video_label = QLabel("INITIALIZING CAMERA...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.video_label.setMinimumSize(640, 480)
        
        # 4. Control Buttons (Bottom)
        self.btn_layout = QHBoxLayout()
        self.stop_btn = QPushButton("STOP")
        self.next_btn = QPushButton(">>")
        self.back_btn = QPushButton("<<")
        self.restart_btn = QPushButton("RESTART")
        
        for btn in [self.stop_btn, self.next_btn, self.back_btn, self.restart_btn]:
            btn.setMinimumHeight(40)
            self.btn_layout.addWidget(btn)

        self.right_col.addWidget(self.video_label, 7) # 70% of right side
        self.right_col.addLayout(self.btn_layout, 3)  # 30% of right side

        # Combine into Main Grid
        self.layout.addLayout(self.left_col, 0, 0)
        self.layout.addLayout(self.right_col, 0, 1)
        self.layout.setColumnStretch(0, 4)
        self.layout.setColumnStretch(1, 6)

        log("SentryHUD Initialized", "INFO")

    def setup_connections(self):
        """ All button logic stays INSIDE the UI class """
        self.stop_btn.clicked.connect(self.handle_stop) # freeze

        self.restart_btn.clicked.connect(self.handle_restart) # clear cache

        self.next_btn.clicked.connect(self.worker.step_forward) # get the next frame when stopped
        self.back_btn.clicked.connect(self.worker.step_backward) # get the previous frame when stopped


    def handle_stop(self):
        self.worker.is_frozen = not self.worker.is_frozen
        
        # Provide visual feedback on the button
        if self.worker.is_frozen:
            self.stop_btn.setText("RESUME")
            self.stop_btn.setStyleSheet("background-color: #444; color: yellow;")
        else:
            self.stop_btn.setText("STOP")
            self.stop_btn.setStyleSheet("background-color: #222; color: white;")

    def handle_restart(self):
        # Clear the history log on the screen
        self.history_list.clear()
        self.worker.reset_tracking_data()

    # ----------- Runs this ^ until moving to the worker init ---------

    def _create_preview_box(self, text):
        lbl = QLabel(text)
        lbl.setFixedSize(112, 112)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        lbl.setStyleSheet("border: 1px solid #555; background-color: #222; color: white; font-size: 10px;")
        return lbl

    def update_displays(self, main_frame, aligned_face, data):
        # 1. Draw the FPS directly on the main_frame (OpenCV BGR format)
        # Positioning at (10, 40) - Top Left
        fps_val = data.get("fps", 0)
        cv2.putText(main_frame, f"FPS: {fps_val}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)


        # Update Main Feed
        rgb_image = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

        # Update Align Preview (The middle small box)
        if aligned_face is not None:
            a_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            a_qt = QImage(a_rgb.data, 112, 112, 112*3, QImage.Format.Format_RGB888)
            self.align_cap.setPixmap(QPixmap.fromImage(a_qt))

        # Update History
        if "history" in data:
            self.history_list.append(data["history"])