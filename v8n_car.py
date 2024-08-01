import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8 OBB 모델 로드
model = YOLO("yolov8n-obb.pt")
names = model.names

# 입력 비디오 파일 열기
input_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\100m\\DJI_0069.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS))

# 비디오 작가 초기화
output_video_path = "v8n_obb_detection.avi"
video_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps, (w, h))

# 차량 클래스 ID 정의
vehicle_classes = [key for key, value in names.items() if value in ['large vehicle', 'small vehicle']]

# 프레임 단위로 비디오 처리
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    # 이미지 크기를 YOLO 모델에 맞게 조정
    im_resized = cv2.resize(im0, (640, 384))  # 모델의 입력 크기에 맞게 조정

    results = model(im_resized)  # 예측 수행
    obb_results = results[0].obb if len(results) > 0 and hasattr(results[0], 'obb') and results[0].obb else None

    if obb_results is not None:
        classes = obb_results.cls.cpu().numpy()
        confidences = obb_results.conf.cpu().numpy()
        boxes = obb_results.xyxyxyxy.cpu().numpy()

        detected_objects = 0
        for cls, conf, box in zip(classes, confidences, boxes):
            if int(cls) in vehicle_classes:
                detected_objects += 1
                vertices = box.reshape(-1, 2)
                pts = (vertices * [w / 640, h / 384]).astype(int)  # 원래 크기에 맞게 좌표 변환
                
                # OBB 바운딩 박스 그리기
                cv2.polylines(im0, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # 클래스 이름과 신뢰도 표시
                label = f"{names[int(cls)]}: {conf:.2f}%"
                label_position = tuple(pts.min(axis=0))
                cv2.putText(im0, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 감지된 객체 수 출력
        print(f"Frame {frame_count}: {detected_objects} objects detected")

    video_writer.write(im0)
    frame_count += 1

# 리소스 해제
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# 동영상 저장 경로 출력
print(f"Detection results saved to {output_video_path}")
