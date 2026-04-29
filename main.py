import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -----------------------
# 허리 각도 계산
# -----------------------
def get_torso_angle(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]
    angle = np.degrees(np.arctan2(dx, dy))
    return angle

# -----------------------
# 상태 판단
# -----------------------
def classify(angle):
    if angle > 20:
        return "FORWARD"
    elif angle < 0:
        return "BACKWARD"
    else:
        return "GOOD"

# -----------------------
# RealSense 설정
# -----------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# -----------------------
# 변수
# -----------------------
good_start_time = None
captured = False

# -----------------------
# 창 설정 
# -----------------------
cv2.namedWindow("Posture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Posture", 1280, 960)

# -----------------------
# 실행
# -----------------------
with mp_pose.Pose(model_complexity=0) as pose:

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        current_time = time.time()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            shoulder_coord = [shoulder.x, shoulder.y]
            hip_coord = [hip.x, hip.y]

            angle = get_torso_angle(shoulder_coord, hip_coord)
            state = classify(angle)

            # -----------------------
            # GOOD 유지 → 촬영
            # -----------------------
            if state == "GOOD":
                if good_start_time is None:
                    good_start_time = current_time

                elapsed = current_time - good_start_time
                remaining = 2 - elapsed

                if remaining > 0:
                    cv2.putText(frame, f"Capture in {int(remaining)+1}",
                                (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)

                elif not captured:
                    cv2.imwrite(f"good_{int(current_time)}.jpg", frame)
                    print("GOOD 촬영!")
                    captured = True

            else:
                good_start_time = None
                captured = False

                # 경고 표시
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

                cv2.putText(frame, "BAD POSTURE!",
                            (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)

            # -----------------------
            # 화면 출력
            # -----------------------
            cv2.putText(frame, f"Angle: {int(angle)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            cv2.putText(frame, f"State: {state}",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Posture", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

pipeline.stop()
cv2.destroyAllWindows()