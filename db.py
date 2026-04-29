import cv2
import numpy as np
import time
import pyrealsense2 as rs
import socket
import struct

from supabase import create_client, Client

# -----------------------
# 🔹 Supabase 연결
# -----------------------
url = "https://zazltktatuloxlsujwiq.supabase.co"
key = "sb_publishable_qo5uKTLC_ghigJElbTKVQA_LIM8yiRP"

supabase: Client = create_client(url, key)

def get_or_create_user_id(name):
    res = supabase.table("users").select("user_id").eq("name", name).execute()

    if len(res.data) > 0:
        return res.data[0]["user_id"]

    new_user = supabase.table("users").insert({
        "name": name
    }).execute()

    return new_user.data[0]["user_id"]

name = input("이름 입력: ")
USER_ID = get_or_create_user_id(name)

print("현재 사용자 ID:", USER_ID)


# -----------------------
# 🔥 소켓 연결 (클라이언트)
# -----------------------
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("127.0.0.1", 7979))

print("✅ 서버 연결 완료")


# -----------------------
# Mediapipe
# -----------------------
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

model_path = "C:/python/capston/pose_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(options)


# -----------------------
# RealSense
# -----------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


# -----------------------
# 함수
# -----------------------
def get_torso_angle(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]
    return np.degrees(np.arctan2(dx, dy))

def classify(angle):
    if angle > 15:
        return "FORWARD"
    elif angle < -10:
        return "BACKWARD"
    else:
        return "GOOD"


# -----------------------
# 변수
# -----------------------
good_start_time = None
captured = False

angle_buffer = []
BUFFER_SIZE = 5


# -----------------------
# 실행
# -----------------------
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    h, w, _ = frame.shape

    current_time = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp = int(current_time * 1000)
    result = detector.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # 🔹 DB용 데이터
        landmark_data = [{"x": float(lm.x), "y": float(lm.y)} for lm in landmarks]

        shoulder = landmarks[11]
        hip = landmarks[23]

        angle = get_torso_angle(
            [shoulder.x, shoulder.y],
            [hip.x, hip.y]
        )

        angle_buffer.append(angle)
        if len(angle_buffer) > BUFFER_SIZE:
            angle_buffer.pop(0)

        angle = np.mean(angle_buffer)
        state = classify(angle)

        # -----------------------
        # GOOD → 저장
        # -----------------------
        if state == "GOOD":
            if good_start_time is None:
                good_start_time = current_time

            if current_time - good_start_time >= 2 and not captured:
                filename = f"good_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame)

                print("📸 촬영 완료:", filename)

                supabase.table("pose_records").insert({
                    "user_id": USER_ID,
                    "landmarks": landmark_data
                }).execute()

                print("✅ DB 저장 완료")

                captured = True
                good_start_time = None

        else:
            good_start_time = None
            captured = False

        cv2.putText(frame, state,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0,255,0), 3)

    # -----------------------
    # 🔥 소켓 전송 (핵심)
    # -----------------------
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    data = buffer.tobytes()

    # 길이 먼저 보내기
    client_socket.sendall(struct.pack(">L", len(data)) + data)

    cv2.imshow("Posture", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
client_socket.close()
cv2.destroyAllWindows()