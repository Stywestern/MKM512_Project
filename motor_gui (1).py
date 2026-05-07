import sys
import threading

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QTextEdit,
    QGroupBox
)

from PoseCtrl import PoseCtrl


IP_ADDRESS = "192.168.0.101"
PORT = 23000

class MotorWorker(QThread):
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, pose_ctrl, command_type, motor1_value=None, motor2_value=None):
        super().__init__()
        self.pose_ctrl = pose_ctrl
        self.command_type = command_type
        self.motor1_value = motor1_value
        self.motor2_value = motor2_value

    def run(self):
        try:
            if self.command_type == "send_pose":
                # PoseCtrl.sendPose negatif değer kabul etmediği için
                # -60 / +60 aralığını 0-359 derece formatına çeviriyoruz.
                motor1_send = self.motor1_value % 360
                motor2_send = self.motor2_value % 360

                result = self.pose_ctrl.sendPose(motor1_send, motor2_send)

            elif self.command_type == "set_f":
                result = self.pose_ctrl.SetF()

            elif self.command_type == "reset_f":
                result = self.pose_ctrl.ResetF()

            else:
                result = ("Error", "Unknown command")

            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))


class MotorControlGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.lock = threading.Lock()
        self.pose_ctrl = PoseCtrl(IP_ADDRESS, PORT, self.lock)

        self.worker = None

        self.setWindowTitle("Motor Control Interface")
        self.setGeometry(300, 300, 500, 420)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # ---------------- CONNECTION GROUP ----------------
        connection_group = QGroupBox("Connection")
        connection_layout = QHBoxLayout()

        self.status_label = QLabel("Status: Not connected")

        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")

        self.connect_button.clicked.connect(self.connect_to_device)
        self.disconnect_button.clicked.connect(self.disconnect_from_device)

        connection_layout.addWidget(self.status_label)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.disconnect_button)

        connection_group.setLayout(connection_layout)

        # ---------------- MOTOR GROUP ----------------
        motor_group = QGroupBox("Motor Position Control")
        motor_layout = QVBoxLayout()

        self.motor1_label = QLabel("Motor 1: 0°")
        self.motor1_slider = QSlider(Qt.Horizontal)
        self.motor1_slider.setMinimum(-60)
        self.motor1_slider.setMaximum(60)
        self.motor1_slider.setValue(0)
        self.motor1_slider.setTickInterval(10)
        self.motor1_slider.setTickPosition(QSlider.TicksBelow)
        self.motor1_slider.valueChanged.connect(self.update_motor1_label)

        self.motor2_label = QLabel("Motor 2: 0°")
        self.motor2_slider = QSlider(Qt.Horizontal)
        self.motor2_slider.setMinimum(-60)
        self.motor2_slider.setMaximum(60)
        self.motor2_slider.setValue(0)
        self.motor2_slider.setTickInterval(10)
        self.motor2_slider.setTickPosition(QSlider.TicksBelow)
        self.motor2_slider.valueChanged.connect(self.update_motor2_label)

        self.send_button = QPushButton("Send to Motors")
        self.send_button.clicked.connect(self.send_motor_values)

        motor_layout.addWidget(self.motor1_label)
        motor_layout.addWidget(self.motor1_slider)
        motor_layout.addWidget(self.motor2_label)
        motor_layout.addWidget(self.motor2_slider)
        motor_layout.addWidget(self.send_button)

        # ---------------- SETF / RESETF BUTTONS ----------------
        fire_button_layout = QHBoxLayout()

        self.set_f_button = QPushButton("SetF")
        self.reset_f_button = QPushButton("ResetF")

        self.set_f_button.clicked.connect(self.set_f)
        self.reset_f_button.clicked.connect(self.reset_f)

        fire_button_layout.addWidget(self.set_f_button)
        fire_button_layout.addWidget(self.reset_f_button)

        motor_layout.addLayout(fire_button_layout)

        motor_group.setLayout(motor_layout)

        # ---------------- LOG BOX ----------------
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        main_layout.addWidget(connection_group)
        main_layout.addWidget(motor_group)
        main_layout.addWidget(QLabel("Log"))
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)

    # ---------------- LABEL UPDATES ----------------
    def update_motor1_label(self):
        value = self.motor1_slider.value()
        self.motor1_label.setText(f"Motor 1: {value}°")

    def update_motor2_label(self):
        value = self.motor2_slider.value()
        self.motor2_label.setText(f"Motor 2: {value}°")

    # ---------------- CONNECTION FUNCTIONS ----------------
    def connect_to_device(self):
        result = self.pose_ctrl.connect()

        if result[0] is True:
            self.status_label.setText("Status: Connected")
            self.log_box.append(f"Connected to {IP_ADDRESS}:{PORT}")
        else:
            self.status_label.setText("Status: Connection failed")
            self.log_box.append(f"Connection error: {result[1]}")
            QMessageBox.critical(self, "Connection Error", result[1])

    def disconnect_from_device(self):
        self.pose_ctrl.disconnect()
        self.status_label.setText("Status: Not connected")
        self.log_box.append("Disconnected.")

    # ---------------- COMMON CHECK ----------------
    def is_device_ready(self):
        if not self.pose_ctrl.Connected:
            QMessageBox.warning(self, "Warning", "Device is not connected.")
            return False

        return True

    # ---------------- MOTOR SEND FUNCTION ----------------
    def send_motor_values(self):
        if not self.is_device_ready():
            return

        motor1_value = self.motor1_slider.value()
        motor2_value = self.motor2_slider.value()

        motor1_send = motor1_value % 360
        motor2_send = motor2_value % 360

        self.log_box.append(
            f"Sending Motor 1: {motor1_value}° -> {motor1_send}, "
            f"Motor 2: {motor2_value}° -> {motor2_send}"
        )

        self.disable_command_buttons()

        self.worker = MotorWorker(
            self.pose_ctrl,
            command_type="send_pose",
            motor1_value=motor1_value,
            motor2_value=motor2_value
        )

        self.worker.finished_signal.connect(self.on_send_finished)
        self.worker.error_signal.connect(self.on_send_error)
        self.worker.start()

    # ---------------- SETF / RESETF FUNCTIONS ----------------
    def set_f(self):
        if not self.is_device_ready():
            return

        self.log_box.append("Sending SetF command...")

        self.disable_command_buttons()

        self.worker = MotorWorker(
            self.pose_ctrl,
            command_type="set_f"
        )

        self.worker.finished_signal.connect(self.on_send_finished)
        self.worker.error_signal.connect(self.on_send_error)
        self.worker.start()

    def reset_f(self):
        if not self.is_device_ready():
            return

        self.log_box.append("Sending ResetF command...")

        self.disable_command_buttons()

        self.worker = MotorWorker(
            self.pose_ctrl,
            command_type="reset_f"
        )

        self.worker.finished_signal.connect(self.on_send_finished)
        self.worker.error_signal.connect(self.on_send_error)
        self.worker.start()

    # ---------------- BUTTON ENABLE / DISABLE ----------------
    def disable_command_buttons(self):
        self.send_button.setEnabled(False)
        self.set_f_button.setEnabled(False)
        self.reset_f_button.setEnabled(False)

    def enable_command_buttons(self):
        self.send_button.setEnabled(True)
        self.set_f_button.setEnabled(True)
        self.reset_f_button.setEnabled(True)

    # ---------------- WORKER RESPONSES ----------------
    def on_send_finished(self, result):
        self.enable_command_buttons()
        self.log_box.append(f"Response: {result}")

    def on_send_error(self, error_message):
        self.enable_command_buttons()
        self.log_box.append(f"Send error: {error_message}")
        QMessageBox.critical(self, "Send Error", error_message)

    # ---------------- CLOSE EVENT ----------------
    def closeEvent(self, event):
        try:
            self.pose_ctrl.disconnect()
        except Exception:
            pass

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MotorControlGUI()
    window.show()
    sys.exit(app.exec_())