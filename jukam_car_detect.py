import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# YOLO 모델 로드
model = YOLO(r"D:\yolo11x-obb.pt")
names = {9: "large vehicle", 10: "small vehicle"}

# 입력 비디오 설정
input_video_path = r"D:\죽암휴게소_고속도로동영상\20250225-Mi3P-Pano-죽암휴계소(하)\DJI_0170.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 열기 실패"

w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
output_video_path = r"D:\죽암휴게소_고속도로동영상\20250225-Mi3P-Pano-죽암휴계소(하)\DJI_0170_output6.MP4"
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ROI 및 중앙선
roi_width, roi_height = 800, 580
roi_x1, roi_y1 = (w - roi_width) // 2, (h - roi_height) // 2
roi_x2, roi_y2 = roi_x1 + roi_width, roi_y1 + roi_height
middle_line_y = (roi_y1 + roi_y2) // 2
line_thickness = 8

# GSD 기준 픽셀당 실제 거리 (1.198cm)
pixel_to_meter = 0.05198
track_history = {}
vehicle_ids = {}
next_id = 0
counted_ids = set()
lane_history = {}
frame_lost_counter = {}

# 누적 속도 저장 리스트
speeds_up = {}
speeds_down = {}

# 속도 계산 함수
def calculate_speed(current_position, previous_position, fps):
    pixel_distance = np.linalg.norm(np.array(current_position) - np.array(previous_position))
    real_distance = pixel_distance * pixel_to_meter
    return (real_distance * fps) * 3.6  # km/h

# 텍스트 박스 표시 함수
def draw_text_with_background(frame, text, position, color, font_scale=0.9):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (50, 50, 50), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    current_ids = set()

    for obb in results.obb:
        class_id = int(obb.cls.cpu().numpy()[0])
        if class_id not in names:
            continue

        obb_coords = obb.xyxyxyxy[0].cpu().numpy().reshape(-1, 2)
        cx, cy = int(np.mean(obb_coords[:, 0])), int(np.mean(obb_coords[:, 1]))

        cv2.polylines(frame, [obb_coords.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
        draw_text_with_background(frame, f"{names[class_id]} {obb.conf.cpu().numpy()[0]:.2f}", (cx, cy - 20), (255, 255, 255))

        # 차량 ID 추적
        matched_id = None
        for vid, (px, py) in vehicle_ids.items():
            if np.linalg.norm((cx - px, cy - py)) < 50:
                matched_id = vid
                break
        if matched_id is None:
            matched_id = next_id
            vehicle_ids[matched_id] = (cx, cy)
            next_id += 1
        else:
            vehicle_ids[matched_id] = (cx, cy)
        current_ids.add(matched_id)
        frame_lost_counter[matched_id] = 0

        # 경로 저장
        if matched_id not in track_history:
            track_history[matched_id] = deque(maxlen=20)
        track_history[matched_id].append((cx, cy))

        # 속도 계산
        speed = 0
        if len(track_history[matched_id]) >= 2:
            prev_pos = track_history[matched_id][-2]
            speed = calculate_speed((cx, cy), prev_pos, fps)

        # 속도 표시: 중앙선 통과한 차량만
        if matched_id in lane_history:
            draw_text_with_background(frame, f"{speed:.1f} km/h", (cx, cy - 40), (0, 255, 0))

        # 중앙선 통과 체크 및 누적 속도 저장
        if matched_id not in counted_ids:
            prev_y = track_history[matched_id][-2][1] if len(track_history[matched_id]) > 1 else cy
            if prev_y < middle_line_y <= cy and cx < (roi_x1 + roi_width // 2):
                counted_ids.add(matched_id)
                lane_history[matched_id] = "down"
                speeds_down[matched_id] = speed
            elif prev_y > middle_line_y >= cy and cx >= (roi_x1 + roi_width // 2):
                counted_ids.add(matched_id)
                lane_history[matched_id] = "up"
                speeds_up[matched_id] = speed

    # 사라진 차량 정리
    for vid in list(vehicle_ids.keys()):
        if vid not in current_ids:
            frame_lost_counter[vid] += 1
            if frame_lost_counter[vid] > 10:
                vehicle_ids.pop(vid, None)
                track_history.pop(vid, None)
                frame_lost_counter.pop(vid, None)

    # 경로 시각화
    for vid, path in track_history.items():
        if vid not in lane_history:
            continue
        color = (0, 0, 255) if lane_history[vid] == "up" else (255, 255, 0)
        for i in range(1, len(path)):
            cv2.line(frame, path[i - 1], path[i], color, 2)

    # 중앙선 표시
    cv2.line(frame, (roi_x1, middle_line_y), (roi_x2, middle_line_y), (255, 255, 255), line_thickness)

    # 평균 속도 계산
    up_count = len(speeds_up)
    down_count = len(speeds_down)
    up_avg = sum(speeds_up.values()) / up_count if up_count else 0
    down_avg = sum(speeds_down.values()) / down_count if down_count else 0

    draw_text_with_background(frame, f"Up: {up_count} | Avg Speed: {up_avg:.1f} km/h", (30, 40), (0, 0, 255))
    draw_text_with_background(frame, f"Down: {down_count} | Avg Speed: {down_avg:.1f} km/h", (30, 80), (255, 255, 0))

    video_writer.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
