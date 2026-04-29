# AI Squat Correction System

AI 기반 스쿼트 자세 교정 시스템입니다.  
카메라를 통해 사용자의 자세를 실시간으로 분석하고, 올바른 자세를 유지하도록 피드백을 제공합니다.

---

## 📌 프로젝트 개요
- 스쿼트 동작을 실시간으로 분석
- 허리 각도를 기반으로 자세 판단
- 올바른 자세(GOOD) 유지 시 자동 캡처

---

## 🛠 사용 기술
- Python
- OpenCV
- MediaPipe
- Intel RealSense
- NumPy

---

## ⚙️ 주요 기능
- 실시간 Pose Detection
- 허리 각도 계산
- 자세 상태 분류 (GOOD / FORWARD / BACKWARD)
- 일정 시간 GOOD 유지 시 자동 캡처

---

## 📊 자세 판별 기준
- FORWARD: 20° 이상
- BACKWARD: -5° 이하
- GOOD: 정상 범위

---

## 🚀 실행 방법
```bash
pip install opencv-python mediapipe pyrealsense2 numpy
python main.py
