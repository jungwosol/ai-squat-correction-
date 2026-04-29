import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

image = cv2.imread("test_2.jpg")

image = cv2.resize(image, (640, 480))

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



with mp_pose.Pose() as pose:
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

cv2.imshow("Pose", image)
cv2.waitKey(0)
cv2.destroyAllWindows()