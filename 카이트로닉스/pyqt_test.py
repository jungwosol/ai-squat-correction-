import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel
)

from PyQt6.QtCore import QTimer

import pyqtgraph as pg

from klib2_python import *


# =====================================
# pyqtgraph 설정
# =====================================
pg.setConfigOptions(imageAxisOrder="row-major")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # =====================================
        # 창 설정
        # =====================================
        self.setWindowTitle("SnowForce Style UI")

        self.resize(1200, 800)

        # =====================================
        # 중앙 위젯
        # =====================================
        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        # =====================================
        # 레이아웃
        # =====================================
        layout = QVBoxLayout()

        central_widget.setLayout(layout)

        # =====================================
        # 상태 표시
        # =====================================
        self.status_label = QLabel(
            "Sensor waiting..."
        )

        layout.addWidget(
            self.status_label
        )

        # =====================================
        # 좌우 체중 표시
        # =====================================
        self.balance_label = QLabel(
            "Left : 0% / Right : 0%"
        )

        layout.addWidget(
            self.balance_label
        )

        # =====================================
        # 자세 상태 표시
        # =====================================
        self.status_analysis = QLabel(
            "STATUS : CENTERED"
        )

        layout.addWidget(
            self.status_analysis
        )

        # =====================================
        # 그래픽 영역
        # =====================================
        self.graphics = pg.GraphicsLayoutWidget()

        layout.addWidget(
            self.graphics
        )

        # =====================================
        # Plot 생성
        # =====================================
        self.plot = self.graphics.addPlot()

        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")

        self.plot.invertY(True)

        self.plot.setXRange(0, 25)
        self.plot.setYRange(0, 22)

        # =====================================
        # 발 이미지 배경
        # =====================================
        self.foot_bg = pg.ImageItem()

        foot_image = plt.imread("foot.png")

        self.foot_bg.setImage(
            foot_image
        )

        self.foot_bg.setOpacity(1.0)

        self.foot_bg.setRect(
            0,
            0,
            25,
            22
        )

        self.plot.addItem(
            self.foot_bg
        )

        # =====================================
        # 압력 점
        # =====================================
        self.pressure_points = pg.ScatterPlotItem()

        self.plot.addItem(
            self.pressure_points
        )

        # =====================================
        # COP 점
        # =====================================
        self.cop_point = pg.ScatterPlotItem(

            size=28,

            brush=pg.mkBrush(
                255,
                255,
                255
            ),

            pen=pg.mkPen(
                color='k',
                width=3
            )
        )

        self.plot.addItem(
            self.cop_point
        )

        # =====================================
        # COP 궤적
        # =====================================
        self.cop_trail = pg.PlotCurveItem(
            pen=pg.mkPen(
                color='r',
                width=3
            )
        )

        self.plot.addItem(
            self.cop_trail
        )

        # COP 좌표 저장
        self.cop_history = []

        # =====================================
        # 센서 연결
        # =====================================
        self.sensor = KLib(
            "127.0.0.1",
            3800
        )

        self.sensor.start()

        print("센서 연결 성공")

        # =====================================
        # FPS 변수
        # =====================================
        self.frame_count = 0

        self.fps_count = 0

        self.last_fps_time = time.time()

        # =====================================
        # 타이머
        # =====================================
        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_sensor
        )

        # 약 50FPS
        self.timer.start(20)

    # =====================================
    # 센서 업데이트
    # =====================================
    def update_sensor(self):

        # =====================================
        # 센서 읽기
        # =====================================
        self.sensor.read()

        # =====================================
        # 배열 변환
        # =====================================
        arr = np.array(
            self.sensor.dataMatrix,
            dtype=np.float32
        )

        arr = arr.reshape((32, 10))

        # =====================================
        # 노이즈 제거
        # =====================================
        arr[arr < 10] = 0

        # =====================================
        # 디버그 출력
        # =====================================
        if self.frame_count % 20 == 0:
            print(np.sum(arr, axis=0))

        # =====================================
        # 압력 점 생성
        # =====================================
        spots = []

        for row in range(32):

            for col in range(10):

                value = arr[row, col]

                if value > 15:

                    # =====================================
                    # 전체 센서를 발 전체에 매핑
                    # =====================================
                    x = col * 2.2 + 2
                    y = row * 0.55 + 2

                    # =====================================
                    # 점 크기
                    # =====================================
                    size = value * 0.6 + 8

                    # =====================================
                    # 색상
                    # =====================================
                    color = pg.intColor(
                        int(value),
                        50
                    )

                    spots.append({

                        'pos': (x, y),

                        'size': size,

                        'brush': color

                    })

        # =====================================
        # 압력 점 업데이트
        # =====================================
        self.pressure_points.setData(
            spots
        )

        # =====================================
        # 좌우 체중 계산
        # =====================================
        left_pressure = np.sum(arr[:, :5])

        right_pressure = np.sum(arr[:, 5:])

        total_lr = left_pressure + right_pressure

        if total_lr > 0:

            left_ratio = (
                left_pressure / total_lr
            ) * 100

            right_ratio = (
                right_pressure / total_lr
            ) * 100

        else:

            left_ratio = 0
            right_ratio = 0

        # =====================================
        # 좌우 체중 표시
        # =====================================
        self.balance_label.setText(
            f"Left : {left_ratio:.1f}%    "
            f"Right : {right_ratio:.1f}%"
        )

        # =====================================
        # COP 계산
        # =====================================
        total_pressure = np.sum(arr)

        if total_pressure > 0:

            rows, cols = np.indices(
                arr.shape
            )

            cop_x_raw = np.sum(
                cols * arr
            ) / total_pressure

            cop_y_raw = np.sum(
                rows * arr
            ) / total_pressure

            # =====================================
            # 화면 좌표 변환
            # =====================================
            cop_x = cop_x_raw * 2.2 + 2
            cop_y = cop_y_raw * 0.55 + 2

            self.cop_point.setData(
                [cop_x],
                [cop_y]
            )

            # =====================================
            # COP 궤적 저장
            # =====================================
            self.cop_history.append(
                (cop_x, cop_y)
            )

            # 최대 길이 제한
            if len(self.cop_history) > 100:
                self.cop_history.pop(0)

            xs = [p[0] for p in self.cop_history]
            ys = [p[1] for p in self.cop_history]

            self.cop_trail.setData(
                xs,
                ys
            )

            # =====================================
            # COP 기반 상태 분석
            # =====================================
            status = "CENTERED"

            # 앞쪽 쏠림
            if cop_y < 7:
                status = "FORWARD SHIFT"

            # 뒤쪽 쏠림
            elif cop_y > 14:
                status = "BACKWARD SHIFT"

            # 왼쪽 쏠림
            if left_ratio > 65:
                status += " / LEFT BIAS"

            # 오른쪽 쏠림
            elif right_ratio > 65:
                status += " / RIGHT BIAS"

            self.status_analysis.setText(
                f"STATUS : {status}"
            )

        else:

            self.cop_point.setData([], [])

        # =====================================
        # 상태 표시
        # =====================================
        self.frame_count += 1

        self.fps_count += 1

        if self.frame_count % 10 == 0:

            max_val = float(
                np.max(arr)
            )

            mean_val = float(
                np.mean(arr)
            )

            total_val = float(
                np.sum(arr)
            )

            self.status_label.setText(
                f"max: {max_val:.1f} / "
                f"mean: {mean_val:.1f} / "
                f"total: {total_val:.1f}"
            )

        # =====================================
        # FPS 출력
        # =====================================
        now = time.time()

        if now - self.last_fps_time >= 1.0:

            print(
                "FPS:",
                self.fps_count
            )

            self.fps_count = 0

            self.last_fps_time = now


# =====================================
# 프로그램 실행
# =====================================
app = QApplication(sys.argv)

window = MainWindow()

window.show()

sys.exit(app.exec())