import sys
import threading
import time
import socket
import struct

import cv2
import numpy as np

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

FPS_TARGET = 30


# -------------------------
# 센서 (압력 데이터)
# -------------------------
class SensorWorker(QObject):
    frame_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            frame = np.random.randint(0, 255, (32, 16)).astype(np.float32)
            self.frame_ready.emit(frame)
            time.sleep(1 / FPS_TARGET)


# -------------------------
# 🔥 카메라 (길이 기반 소켓 방식)
# -------------------------
class CameraWorker(QObject):
    frame_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False

    def _run(self):
        HOST = "0.0.0.0"
        PORT = 7979

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)

        print(f"[CameraWorker] Listening on {PORT}...")
        conn, addr = server_socket.accept()
        print(f"[CameraWorker] Connected: {addr}")

        data_buffer = b""
        payload_size = struct.calcsize(">L")

        while self._running:
            try:
                # -----------------------
                # 1️⃣ 길이 받기
                # -----------------------
                while len(data_buffer) < payload_size:
                    packet = conn.recv(4096)
                    if not packet:
                        return
                    data_buffer += packet

                packed_size = data_buffer[:payload_size]
                data_buffer = data_buffer[payload_size:]
                frame_size = struct.unpack(">L", packed_size)[0]

                # -----------------------
                # 2️⃣ 프레임 데이터 받기
                # -----------------------
                while len(data_buffer) < frame_size:
                    packet = conn.recv(4096)
                    if not packet:
                        return
                    data_buffer += packet

                frame_data = data_buffer[:frame_size]
                data_buffer = data_buffer[frame_size:]

                # -----------------------
                # 3️⃣ 디코딩
                # -----------------------
                np_data = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape

                qimg = QImage(
                    rgb.data,
                    w,
                    h,
                    ch * w,
                    QImage.Format.Format_RGB888
                )

                self.frame_ready.emit(qimg.copy())

            except Exception as e:
                print("Socket Error:", e)
                break

        conn.close()
        server_socket.close()


# -------------------------
# 카메라 UI
# -------------------------
class CameraWidget(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.setStyleSheet("""
            background:#f5f5f5;
            border:1px solid #ddd;
        """)

    def update_frame(self, qimg):
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)


# -------------------------
# 측정 탭
# -------------------------
class MeasureTab(QWidget):
    def __init__(self):
        super().__init__()
        self._latest_pressure = None

        self.sensor = SensorWorker()
        self.camera = CameraWorker()

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        self.btn = QPushButton("측정 시작")
        self.btn.setFixedHeight(40)
        self.btn.clicked.connect(self.toggle)
        root.addWidget(self.btn)

        content = QHBoxLayout()

        # 왼쪽 (압력)
        left_panel = QVBoxLayout()

        lbl_pressure = QLabel("발바닥 압력")
        lbl_pressure.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.pressure_placeholder = QLabel("압력 데이터 대기중...")
        self.pressure_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pressure_placeholder.setFixedSize(640, 480)

        left_panel.addWidget(lbl_pressure)
        left_panel.addWidget(self.pressure_placeholder)

        # 오른쪽 (카메라)
        right_panel = QVBoxLayout()

        lbl_camera = QLabel("카메라 영상")
        lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.camera_view = CameraWidget()

        right_panel.addWidget(lbl_camera)
        right_panel.addWidget(self.camera_view)

        content.addLayout(left_panel)
        content.addLayout(right_panel)

        root.addLayout(content)

        self.sensor.frame_ready.connect(self._on_pressure)
        self.camera.frame_ready.connect(self.camera_view.update_frame)

    def toggle(self):
        if self.btn.text() == "측정 시작":
            self.sensor.start()
            self.camera.start()
            self.btn.setText("중지")
        else:
            self.sensor.stop()
            self.camera.stop()
            self.btn.setText("측정 시작")

    def _on_pressure(self, frame):
        self._latest_pressure = frame

    def get_latest_pressure_snapshot(self):
        return self._latest_pressure


# -------------------------
# AI 탭
# -------------------------
class AITab(QWidget):
    def __init__(self, provider):
        super().__init__()
        self.provider = provider

        layout = QVBoxLayout(self)

        self.btn = QPushButton("AI 분석")
        self.result = QTextEdit()

        self.btn.clicked.connect(self.run_ai)

        layout.addWidget(self.btn)
        layout.addWidget(self.result)

    def run_ai(self):
        data = self.provider()
        if data is None:
            self.result.setText("데이터 없음")
            return

        self.result.setText(f"평균 압력: {data.mean():.2f}")


# -------------------------
# 메인
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1200, 700)

        self.tabs = QTabWidget()

        self.measure = MeasureTab()
        self.ai = AITab(self.measure.get_latest_pressure_snapshot)

        self.tabs.addTab(self.measure, "측정")
        self.tabs.addTab(self.ai, "AI")

        self.setCentralWidget(self.tabs)
        self.setStatusBar(QStatusBar())


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()