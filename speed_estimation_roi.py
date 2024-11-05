import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import os

# YOLO 모델 로드
model = YOLO(r"D:\yolo11l-obb.pt")  # YOLO 모델 경로
names = {9: "large vehicle", 10: "small vehicle"}

# 입력 비디오 파일 설정
input_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\DJI_0076.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

# 결과 비디오 저장 설정
output_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\obb_detection_with_speed_estimation_tile_roi.avi"
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 50프레임마다 캡처한 프레임을 저장할 폴더
capture_folder = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\capture_frames3"
os.makedirs(capture_folder, exist_ok=True)

# 중앙 ROI 영역 설정
roi_width, roi_height = 640, 480  # 중앙 영역 크기
roi_x1, roi_y1 = (w - roi_width) // 2, (h - roi_height) // 2
roi_x2, roi_y2 = roi_x1 + roi_width, roi_y1 + roi_height

# 색상 설정
colors = {"small vehicle": (255, 0, 0), "large vehicle": (0, 0, 255)}
speed_color = (0, 255, 0)
roi_color = (255, 0, 255)  # 보라색 ROI

# 차량 위치 및 속도를 추적하기 위한 딕셔너리 초기화
previous_positions = {}
velocity_history = {}

# 정지 상태 판단을 위한 임계값 설정
MIN_POSITION_CHANGE_THRESHOLD = 5  # 최소 위치 변화 임계값 (픽셀)

# 속도 계산 함수
def calculate_speed(current_position, previous_position, fps):
    distance = np.sqrt((current_position[0] - previous_position[0]) ** 2 +
                       (current_position[1] - previous_position[1]) ** 2)
    if distance < MIN_POSITION_CHANGE_THRESHOLD:
        return 0  # 이동 거리가 작을 경우 속도를 0으로 간주
    speed_kmh = (distance * fps) * 0.036  # 픽셀 거리 -> km/h로 변환
    return speed_kmh

# 타일 기반 예측 및 표시 함수
def process_and_annotate_tile(frame, x, y, tile_size=(640, 640)):
    tile = frame[y:y + tile_size[1], x:x + tile_size[0]]  # 타일 추출
    results = model(tile)
    annotated_tile = tile.copy()
    small_vehicle_count, large_vehicle_count = 0, 0

    if results[0].obb:
        for i, obb in enumerate(results[0].obb):
            class_id = int(obb.cls.cpu().numpy()[0])

            # 9(large vehicle)와 10(small vehicle) 클래스만 표시
            if class_id not in names:
                continue

            obb_coords = obb.xyxyxyxy[0].cpu().numpy().reshape(-1, 2)
            x_coords, y_coords = obb_coords[:, 0], obb_coords[:, 1]
            cx, cy = np.mean(x_coords) + x, np.mean(y_coords) + y  # 타일 내 위치에서 전체 프레임 내 위치로 변환

            class_name = names[class_id]
            confidence = float(obb.conf.cpu().numpy()[0])
            color = colors[class_name]

            # 차량 카운트
            if class_id == 9:
                large_vehicle_count += 1
            elif class_id == 10:
                small_vehicle_count += 1

            # 중앙 ROI 내에서의 속도 추정
            if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                vehicle_id = f"vehicle_{i}_{x}_{y}"
                if vehicle_id not in previous_positions:
                    previous_positions[vehicle_id] = deque(maxlen=2)
                    velocity_history[vehicle_id] = deque(maxlen=5)

                # 이전 위치 추가 및 속도 계산
                previous_positions[vehicle_id].append((cx, cy))
                if len(previous_positions[vehicle_id]) > 1:
                    prev_position = previous_positions[vehicle_id][-2]
                    speed = calculate_speed((cx, cy), prev_position, fps)

                    # 속도 표시
                    if speed is not None:
                        velocity_history[vehicle_id].append(speed)
                        avg_speed = np.mean(velocity_history[vehicle_id])
                        cv2.putText(frame, f"{avg_speed:.2f} km/h", (int(cx), int(cy) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)

            # OBB 시각화 및 클래스명과 정확도 표시
            pts = (obb_coords + [x, y]).astype(int)  # 전체 프레임 위치로 좌표 변환
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (int(cx), int(cy) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, small_vehicle_count, large_vehicle_count

# 전체 프레임에 대해 타일별로 탐지 수행
def process_frame_by_tiles(frame):
    total_small_vehicles, total_large_vehicles = 0, 0

    for y in range(0, h, 640):
        for x in range(0, w, 640):
            frame, small_vehicle_count, large_vehicle_count = process_and_annotate_tile(frame, x, y)
            total_small_vehicles += small_vehicle_count
            total_large_vehicles += large_vehicle_count

    # 우측 상단에 차량 개수 카운팅 표시
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 200, 10), (w - 10, 80), (50, 50, 50), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, f"Small Vehicles: {total_small_vehicles}", (w - 190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Large Vehicles: {total_large_vehicles}", (w - 190, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ROI 영역 표시
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)  # 보라색 ROI를 화면에 표시

    return frame

# 프레임별로 비디오 처리
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = process_frame_by_tiles(frame)

    # 결과 비디오 저장
    video_writer.write(annotated_frame)

    # 50 프레임마다 캡처하여 이미지 저장
    if frame_count % 50 == 0:
        capture_path = os.path.join(capture_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(capture_path, annotated_frame)

    # 결과 프레임을 표시
    cv2.imshow("Vehicle Detection with Speed Estimation in Center ROI", annotated_frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Detection results saved to {output_video_path}")
print(f"Captured frames saved to {capture_folder}")
