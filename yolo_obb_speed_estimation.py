import cv2
from ultralytics import YOLO
import numpy as np
import os

# YOLO 모델 로드
model = YOLO(r"D:\yolo11l-obb.pt")  # 모델 경로
names = {9: "large vehicle", 10: "small vehicle"}

# 입력 비디오 파일 설정
input_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\DJI_0076.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

# 출력 비디오 저장 경로 및 파일 이름 설정
output_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\obb_detection_200m_yolo11_speed3.avi"
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

# 타일 크기 설정
tile_size = (640, 640)

# 캡처 프레임 폴더 설정
capture_folder = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\capture_frames2"
os.makedirs(capture_folder, exist_ok=True)

# 프레임 간 차량 위치를 저장하여 속도 추정을 위한 딕셔너리 초기화
previous_positions = {}

# 색상 설정
colors = {"small vehicle": (255, 0, 0), "large vehicle": (0, 0, 255)}  # 파란색과 빨간색

# 속도 계산 함수
def calculate_speed(current_position, previous_position, fps):
    distance = np.sqrt((current_position[0] - previous_position[0]) ** 2 + (current_position[1] - previous_position[1]) ** 2)
    if distance < 5:  # 이동 거리가 5 픽셀 이하일 경우 속도 표시 생략
        return None
    speed_kmh = (distance * fps) * 0.036  # 픽셀 거리 -> km/h
    return speed_kmh

# 타일 기반 예측 및 표시 함수
def process_and_annotate_tile(frame, x, y, tile_id):
    tile = frame[y:y + tile_size[1], x:x + tile_size[0]]  # 타일 생성
    results = model(tile)

    if results[0].obb:
        for i, obb in enumerate(results[0].obb):
            class_id = int(obb.cls.cpu().numpy()[0])

            # 9(large vehicle)와 10(small vehicle) 클래스만 표시
            if class_id not in names:
                continue

            obb_coords = obb.xyxyxyxy[0].cpu().numpy().reshape(-1, 2)
            x_coords, y_coords = obb_coords[:, 0], obb_coords[:, 1]
            cx, cy = np.mean(x_coords), np.mean(y_coords)

            class_name = names[class_id]
            confidence = float(obb.conf.cpu().numpy()[0])
            color = colors[class_name]

            # 속도 추정
            vehicle_id = f"tile_{tile_id}_vehicle_{i}"
            if vehicle_id in previous_positions:
                prev_cx, prev_cy = previous_positions[vehicle_id]
                speed = calculate_speed((cx, cy), (prev_cx, prev_cy), fps)
                if speed is not None:  # 속도가 None이 아닌 경우에만 표시
                    cv2.putText(tile, f"{speed:.2f} km/h", (int(cx), int(cy) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            previous_positions[vehicle_id] = (cx, cy)

            # OBB 시각화
            pts = obb_coords.astype(int)
            cv2.polylines(tile, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(tile, f"{class_name} {confidence:.2f}", (int(cx), int(cy) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return tile

# 프레임별로 비디오 처리
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 타일 단위로 프레임을 분할하여 예측 수행
    tile_id = 0  # 타일 ID 초기화
    for y in range(0, h, tile_size[1]):
        for x in range(0, w, tile_size[0]):
            annotated_tile = process_and_annotate_tile(frame, x, y, tile_id)
            x_end, y_end = min(x + tile_size[0], w), min(y + tile_size[1], h)
            frame[y:y_end, x:x_end] = annotated_tile  # 원본 프레임에 타일 병합
            tile_id += 1

    # 50 프레임마다 캡처하여 저장
    if frame_count % 50 == 0:
        capture_path = os.path.join(capture_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(capture_path, frame)

    # 비디오에 현재 프레임 저장
    video_writer.write(frame)
    frame_count += 1

    # 화면에 결과를 표시
    cv2.imshow('Car Detection with Speed Estimation in Tiles', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"Detection results saved to {output_video_path}")








