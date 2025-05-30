import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import os

# YOLO 모델 로드
model = YOLO(r"D:\yolo11l-obb.pt")
names = {9: "large vehicle", 10: "small vehicle"}

# 입력 비디오 파일 설정
input_video_path = r"C:\Users\dromii\20240401-송도\M3C\300m\DJI_0082.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

# 결과 비디오 저장 설정
output_video_path = r"C:\Users\dromii\20240401-송도\M3C\300m\DJI_0082_300m_SHAI_fixed.MP4"
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ROI 설정
roi_width, roi_height = 800, 480
roi_x1, roi_y1 = (w - roi_width) // 2, (h - roi_height) // 2
roi_x2, roi_y2 = roi_x1 + roi_width, roi_y1 + roi_height

# 타일 크기 설정 (SHAI 적용)
tile_size = (640, 640)

# 색상 설정
colors = {"small vehicle": (255, 182, 193), "large vehicle": (176, 224, 230)}
speed_color = (0, 255, 127)

# 차량 추적 초기화
track_history = {}
vehicle_ids = {}
next_id = 0

# 속도 계산 함수
def calculate_speed(current_position, previous_position, fps):
    distance = np.linalg.norm(np.array(current_position) - np.array(previous_position))
    speed_kmh = (distance * fps) * 0.036
    return min(speed_kmh, 50)

# OBB 중심점 계산 함수
def calculate_center(obb_coords):
    x_coords = obb_coords[:, 0]
    y_coords = obb_coords[:, 1]
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)
    return int(cx), int(cy)

# 텍스트와 반투명 배경 표시 함수
def draw_text_with_background(frame, text, position, color):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (50, 50, 50), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 차량 ID 매칭 함수 (강화된 로직)
def match_vehicle_id(cx, cy, obb_coords, threshold=50):
    global next_id
    for vehicle_id, (prev_cx, prev_cy, prev_obb) in vehicle_ids.items():
        distance = np.linalg.norm([cx - prev_cx, cy - prev_cy])
        obb_similarity = np.linalg.norm(prev_obb - obb_coords)
        if distance < threshold and obb_similarity < threshold:
            vehicle_ids[vehicle_id] = (cx, cy, obb_coords)
            return vehicle_id
    vehicle_id = next_id
    next_id += 1
    vehicle_ids[vehicle_id] = (cx, cy, obb_coords)
    return vehicle_id

# 차량 경로를 그리는 함수
def draw_vehicle_path(frame, vehicle_id, center_point):
    if vehicle_id not in track_history:
        track_history[vehicle_id] = deque(maxlen=200)
    track_history[vehicle_id].append(center_point)

    for i in range(1, len(track_history[vehicle_id])):
        pt1 = track_history[vehicle_id][i - 1]
        pt2 = track_history[vehicle_id][i]
        if roi_x1 <= pt1[0] <= roi_x2 and roi_y1 <= pt1[1] <= roi_y2 and roi_x1 <= pt2[0] <= roi_x2 and roi_y1 <= pt2[1] <= roi_y2:
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

# 프레임별 비디오 처리
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for y in range(0, h, tile_size[1]):
        for x in range(0, w, tile_size[0]):
            tile = frame[y:y + tile_size[1], x:x + tile_size[0]]
            results = model(tile)
            x_offset, y_offset = x, y

            for i, obb in enumerate(results[0].obb):
                class_id = int(obb.cls.cpu().numpy()[0])
                if class_id not in names:
                    continue

                obb_coords = obb.xyxyxyxy[0].cpu().numpy().reshape(-1, 2)
                obb_coords[:, 0] += x_offset
                obb_coords[:, 1] += y_offset
                cx, cy = calculate_center(obb_coords)
                class_name = names[class_id]
                confidence = float(obb.conf.cpu().numpy()[0])
                color = colors[class_name]

                cv2.polylines(frame, [obb_coords.astype(int)], isClosed=True, color=color, thickness=2)
                label_text = f"{class_name} {confidence:.2f}"
                draw_text_with_background(frame, label_text, (cx, cy - 40), color)

                vehicle_id = match_vehicle_id(cx, cy, obb_coords)
                draw_vehicle_path(frame, vehicle_id, (cx, cy))

                if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                    if vehicle_id not in track_history or len(track_history[vehicle_id]) < 2:
                        continue
                    prev_position = track_history[vehicle_id][-2]
                    speed = calculate_speed((cx, cy), prev_position, fps)
                    speed_text = f"{speed:.2f} km/h"
                    draw_text_with_background(frame, speed_text, (cx, cy - 20), speed_color)

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (138, 43, 226), 2)

    video_writer.write(frame)
    cv2.imshow("OBB Detection with SHAI", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Detection and tracking results saved to {output_video_path}")

