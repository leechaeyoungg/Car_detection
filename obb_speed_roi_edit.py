import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import os

# YOLO 모델 로드
model = YOLO(r"D:\yolo11l-obb.pt")
names = {9: "large vehicle", 10: "small vehicle"}

# 입력 비디오 파일 설정
input_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\DJI_0076.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

# 결과 비디오 저장 설정
output_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\obb_detection_final.avi"
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 캡처 폴더 설정
capture_folder = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\capture_frames_final"
os.makedirs(capture_folder, exist_ok=True)

# ROI 설정
roi_width, roi_height = 800, 480
roi_x1, roi_y1 = (w - roi_width) // 2, (h - roi_height) // 2
roi_x2, roi_y2 = roi_x1 + roi_width + 100, roi_y1 + roi_height

# 색상 설정 (파스텔 톤)
colors = {"small vehicle": (255, 182, 193), "large vehicle": (176, 224, 230)}
speed_color = (0, 255, 127)

# SHAI 타일 크기 설정
tile_size = (640, 640)

# 차량 위치 및 속도 추적 초기화
previous_positions = {}
velocity_history = {}

# 속도 계산 함수 (최대 속도 50km/h로 제한)
def calculate_speed(current_position, previous_position, fps):
    distance = np.linalg.norm(np.array(current_position) - np.array(previous_position))
    speed_kmh = (distance * fps) * 0.036

    # 비정상적으로 높은 속도를 제한 (최대 50km/h)
    if speed_kmh > 50:
        speed_kmh = 50

    return speed_kmh


# 텍스트와 반투명 배경 표시 함수
def draw_text_with_background(frame, text, position, color):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = position
    # 반투명 배경
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (50, 50, 50), -1)
    # 텍스트 표시
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 타일 기반 예측 함수
def process_tile(frame, x, y):
    tile = frame[y:y + tile_size[1], x:x + tile_size[0]]
    results = model(tile)
    return results

# OBB 시각화 및 속도 표시 함수
def process_and_annotate_frame(frame):
    small_vehicle_count = 0
    large_vehicle_count = 0

    for y in range(0, h, tile_size[1]):
        for x in range(0, w, tile_size[0]):
            results = process_tile(frame, x, y)

            if results[0].obb:
                for i, obb in enumerate(results[0].obb):
                    class_id = int(obb.cls.cpu().numpy()[0])

                    if class_id not in names:
                        continue

                    try:
                        # OBB 좌표 추출 및 전체 프레임 좌표로 변환
                        obb_coords = obb.xyxyxyxy[0].cpu().numpy().reshape(-1, 2)
                        obb_coords[:, 0] += x
                        obb_coords[:, 1] += y
                        x_coords, y_coords = obb_coords[:, 0], obb_coords[:, 1]
                        cx, cy = np.mean(x_coords), np.mean(y_coords)

                        class_name = names[class_id]
                        confidence = float(obb.conf.cpu().numpy()[0])
                        color = colors[class_name]

                        # 차량 카운트
                        if class_id == 9:
                            large_vehicle_count += 1
                        elif class_id == 10:
                            small_vehicle_count += 1

                        # ROI 내에서만 속도 표시
                        if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                            vehicle_id = f"vehicle_{i}_{x}_{y}"
                            if vehicle_id not in previous_positions:
                                previous_positions[vehicle_id] = deque(maxlen=2)
                                velocity_history[vehicle_id] = deque(maxlen=5)

                            previous_positions[vehicle_id].append((cx, cy))
                            if len(previous_positions[vehicle_id]) > 1:
                                prev_position = previous_positions[vehicle_id][-2]
                                speed = calculate_speed((cx, cy), prev_position, fps)
                                velocity_history[vehicle_id].append(speed)
                                avg_speed = np.mean(velocity_history[vehicle_id])

                                speed_text = f"{avg_speed:.2f} km/h"
                                draw_text_with_background(frame, speed_text, (int(cx), int(cy) - 20), speed_color)

                        # 클래스명과 정확도 표시
                        label_text = f"{class_name} {confidence:.2f}"
                        draw_text_with_background(frame, label_text, (int(cx), int(cy) - 40), color)

                        # OBB를 타원 형태로 시각화
                        if len(obb_coords) >= 5:
                            ellipse = cv2.fitEllipse(obb_coords.astype(np.int32))
                            cv2.ellipse(frame, ellipse, color, 2)
                        else:
                            pts = obb_coords.astype(int)
                            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

                    except Exception as e:
                        print(f"Error processing OBB: {e}")

    # 차량 개수 표시
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 220, 10), (w - 10, 80), (50, 50, 50), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, f"Small Vehicles: {small_vehicle_count}", (w - 210, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Large Vehicles: {large_vehicle_count}", (w - 210, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ROI 표시
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (138, 43, 226), 4)

    return frame

# 프레임별 비디오 처리
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = process_and_annotate_frame(frame)

    video_writer.write(annotated_frame)

    if frame_count % 50 == 0:
        capture_path = os.path.join(capture_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(capture_path, annotated_frame)

    cv2.imshow("OBB Detection with SHAI", annotated_frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Detection results saved to {output_video_path}")
print(f"Captured frames saved to {capture_folder}")
