import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8 OBB 모델 로드
model = YOLO(r"D:\yolo11l-obb.pt")
names = model.names

# 입력 비디오 파일 열기
input_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\DJI_0076.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS))

# 비디오 작가 초기화
output_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\obb_detection_200m_yolo11_shai.avi"
video_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps, (w, h))

# 타일 크기
tile_size = 240

# 차량 클래스 ID 정의
vehicle_classes = ['large vehicle', 'small vehicle']

# 클래스별 색상 정의
class_colors = {
    'large vehicle': (0, 0, 255),  # 빨간색
    'small vehicle': (0, 255, 0),  # 초록색
}

# 투명도를 위한 함수
def add_transparent_textbox(frame, text, position, font_scale=2.0, font_thickness=3, padding=15):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    box_w, box_h = text_size
    x, y = position
    box_coords = ((x, y), (x + box_w + padding * 2, y - box_h - padding * 2))
    overlay = frame.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], (50, 50, 50), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, text, (x + padding, y - padding), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    return frame

# 타일링 및 예측 함수
def process_tile(tile, x_offset, y_offset):
    results = model(tile)
    obb_results = results[0].obb if len(results) > 0 and hasattr(results[0], 'obb') and results[0].obb else None

    detections = []
    if obb_results is not None:
        classes = obb_results.cls.cpu().numpy()
        confidences = obb_results.conf.cpu().numpy()
        boxes = obb_results.xyxyxyxy.cpu().numpy()

        for cls, conf, box in zip(classes, confidences, boxes):
            cls_name = names[int(cls)]
            if cls_name in vehicle_classes:
                # 바운딩 박스 좌표 변환
                vertices = box.reshape(-1, 2)
                pts = (vertices * [tile.shape[1] / 640, tile.shape[0] / 384]).astype(int)
                pts += np.array([x_offset, y_offset])  # 타일 좌표에 맞춰 조정
                detections.append((cls_name, conf, pts))
    return detections

# 프레임 단위로 비디오 처리
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    class_count = {cls: 0 for cls in vehicle_classes}

    # 프레임을 타일 크기 기준으로 나누어 예측 수행
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = im0[y:y+tile_size, x:x+tile_size]
            detections = process_tile(tile, x, y)

            for cls_name, conf, pts in detections:
                class_count[cls_name] += 1
                color = class_colors[cls_name]
                cv2.polylines(im0, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                label = f"{cls_name}: {conf:.2f}%"
                label_position = tuple(pts.min(axis=0))
                cv2.putText(im0, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)

    count_text = " | ".join([f"{cls}: {count}" for cls, count in class_count.items()])
    im0 = add_transparent_textbox(im0, count_text, (10, 60), font_scale=2.0, font_thickness=3)

    # 감지된 객체 수 출력 (프레임별 출력)
    print(f"Frame {frame_count}: {class_count}")

    video_writer.write(im0)
    frame_count += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Detection results saved to {output_video_path}")


