from klib2_python import *

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 센서 설정
# =========================
HOST = "127.0.0.1"
PORT = 3800

ROWS = 32
COLS = 10

PRESSURE_MAX = 50
NOISE_THRESHOLD = 3

# =========================
# 센서 연결
# =========================
sensor = KLib(HOST, PORT)

sensor.start()

print("센서 연결 성공")

# =========================
# matplotlib 설정
# =========================
plt.ion()

fig, ax = plt.subplots(figsize=(6, 10))

# 초기 데이터
dummy = np.zeros((ROWS, COLS))

# Heatmap 생성
heatmap = ax.imshow(
    dummy,
    cmap='jet',
    vmin=0,
    vmax=PRESSURE_MAX,
    interpolation='nearest',
    aspect='auto'
)

# COP 점
cop_point, = ax.plot(
    [],
    [],
    'ro',
    markersize=10
)

# 제목
ax.set_title("Foot Pressure Heatmap")

# 구분선 추가
ax.axhline(
    y=15.5,
    color='white',
    linewidth=2
)

# 텍스트 표시
ax.text(
    0,
    2,
    "LEFT FOOT",
    color='white',
    fontsize=14,
    weight='bold'
)

ax.text(
    0,
    18,
    "RIGHT FOOT",
    color='white',
    fontsize=14,
    weight='bold'
)

# 컬러바
plt.colorbar(heatmap)

# =========================
# 실시간 루프
# =========================
try:

    while True:

        # 센서 읽기
        sensor.read()

        # 배열 변환
        arr = np.array(
            sensor.dataMatrix,
            dtype=np.float32
        )

        # 중요!!
        arr = arr.reshape((ROWS, COLS), order='F')

        # 노이즈 제거
        arr[arr < NOISE_THRESHOLD] = 0

        # =========================
        # 방향 조정
        # =========================

        # 위 16줄 = 왼발
        left_foot = arr[16:, :]

        # 아래 16줄 = 오른발
        right_foot = arr[:16, :]

        # 합치기
        final_map = np.vstack([
            left_foot,
            right_foot
        ])

        # =========================
        # Heatmap 업데이트
        # =========================
        heatmap.set_data(final_map)

        # =========================
        # COP 계산
        # =========================
        total_pressure = np.sum(final_map)

        if total_pressure > 0:

            rows, cols = np.indices(final_map.shape)

            cop_x = np.sum(cols * final_map) / total_pressure

            cop_y = np.sum(rows * final_map) / total_pressure

            cop_point.set_data(
                [cop_x],
                [cop_y]
            )

        else:

            cop_point.set_data([], [])

        # =========================
        # 화면 갱신
        # =========================
        fig.canvas.draw_idle()

        fig.canvas.flush_events()

        plt.pause(0.01)

except KeyboardInterrupt:

    print("프로그램 종료")