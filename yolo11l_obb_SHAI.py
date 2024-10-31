import cv2
from ultralytics import YOLO
import numpy as np
import os

# YOLOv8 모델 로드
model = YOLO(r"D:\yolo11l-obb.pt")  # 모델 경로
names = model.names

# 입력 비디오 파일 설정
input_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\DJI_0076.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

# 비디오 저장 설정
output_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\obb_detection_200m_yolo11_shai_640.avi"
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

# 타일 크기 설정
tile_size = (640, 640)

# 캡처 프레임 폴더 설정
capture_folder = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\capture_frames"
os.makedirs(capture_folder, exist_ok=True)

# 타일 기반 예측 및 표시 함수
def process_and_annotate_tile(frame, x, y):
    tile = frame[y:y + tile_size[1], x:x + tile_size[0]]  # 타일 생성
    results = model(tile)  # 모델 적용
    annotated_tile = results[0].plot()  # 예측된 결과를 타일에 표시
    return annotated_tile

# 프레임별로 비디오 처리
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 타일 단위로 프레임 분할 및 예측 수행
    for y in range(0, h, tile_size[1]):
        for x in range(0, w, tile_size[0]):
            annotated_tile = process_and_annotate_tile(frame, x, y)
            x_end, y_end = min(x + tile_size[0], w), min(y + tile_size[1], h)
            frame[y:y_end, x:x_end] = annotated_tile  # 원본 프레임에 표시된 타일 병합

    # 중간 프레임 저장 (50 프레임마다 캡처)
    if frame_count % 50 == 0:
        capture_path = os.path.join(capture_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(capture_path, frame)

    # 비디오에 현재 프레임 저장
    video_writer.write(frame)
    frame_count += 1

    # 화면에 결과를 표시
    cv2.imshow('Bird Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"Detection results saved to {output_video_path}")






