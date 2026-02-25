# modules/interface.py

##################################### Imports #####################################
# Libraries
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2

# Modules
import config
from modules.utils import log

###################################################################################

class SentryHUD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentry Turret V1 - Debug HUD")
        self.init_ui()

    def init_ui(self):
        # Main Container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Horizontal layout: [ Left: Control/Data | Right: 40% Camera ]
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- LEFT PANEL (60%) ---
        self.left_panel = QVBoxLayout()
        
            # Snapshot displays (Aligned Face Preview)
        self.aligned_preview = QLabel("Aligned Face")
        self.aligned_preview.setFixedSize(112, 112)
        self.aligned_preview.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        self.left_panel.addWidget(self.aligned_preview)

            # Buttons
        self.freeze_btn = QPushButton("FREEZE SYSTEM")
        self.reset_btn = QPushButton("RESET TRACKING")
        self.left_panel.addWidget(self.freeze_btn)
        self.left_panel.addStretch()

        # --- RIGHT PANEL (40%) ---
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        
            # Add to main layout
        self.main_layout.addLayout(self.left_panel, 6)
        self.main_layout.addWidget(self.video_label, 4)

        log("SentryHUD Initialized", "INFO")

    # ----------- Runs this ^ until moving to the worker init ---------

    def update_main_feed(self, cv_img, aligned_face, data):
        # 1. Update the Text Stats
        fps_val = data.get("fps", 0)
        cv2.putText(cv_img, f"FPS: {fps_val}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))