import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort  # SORT 모듈 임포트

# IoU 계산 함수
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

# YOLOv8 OBB 모델 로드
model = YOLO("yolov8l-obb.pt")
names = model.names

# SORT 추적기 초기화
tracker = Sort()

# 입력 비디오 파일 열기
input_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\100m\\DJI_0069.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    int(cap.get(cv2.CAP_PROP_FPS))
)

# 비디오 작가 초기화
output_video_path = "AABB_IoU_applied.avi"
video_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps, (w, h)
)

# 차량 클래스 정의
vehicle_classes = ['large vehicle', 'small vehicle']

# 클래스별 색상 정의
class_colors = {
    'large vehicle': (0, 0, 255),  # 빨간색
    'small vehicle': (0, 255, 0),  # 초록색
}


#투명도 함수
def add_transparent_textbox(frame, text, position, font_scale=2.0, font_thickness=3, padding=15):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    box_w, box_h = text_size

    # 투명한 텍스트박스 좌표 계산
    x, y = position
    box_coords = ((x, y), (x + box_w + padding * 2, y - box_h - padding * 2))
    overlay = frame.copy()

    # 회색 투명 텍스트박스 추가
    cv2.rectangle(overlay, box_coords[0], box_coords[1], (50, 50, 50), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # 텍스트 추가
    cv2.putText(frame, text, (x + padding, y - padding), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    return frame

# 프레임 단위로 비디오 처리
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    # 모델 예측 수행
    results = model(im0, imgsz=1024)
    obb_results = results[0].obb if len(results) > 0 and hasattr(results[0], 'obb') and results[0].obb else None

    detections = []
    class_mappings = []

    # 클래스별 카운팅을 위한 딕셔너리
    class_count = {cls: 0 for cls in vehicle_classes}

    if obb_results is not None:
        classes = obb_results.cls.cpu().numpy()
        confidences = obb_results.conf.cpu().numpy()
        boxes = obb_results.xyxy.cpu().numpy()

        for cls, conf, box in zip(classes, confidences, boxes):
            cls_name = names[int(cls)]
            if cls_name in vehicle_classes:
                class_count[cls_name] += 1
                x1, y1, x2, y2 = box.astype(int)
                detections.append([x1, y1, x2, y2, conf])
                class_mappings.append({'bbox': (x1, y1, x2, y2), 'class': cls_name, 'conf': conf})

    # 추적기 업데이트
    tracked_objects = tracker.update(np.array(detections))

    # 추적된 객체에 대해 바운딩 박스 그리기
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)

        # 추적된 객체와 가장 높은 IoU를 갖는 감지된 객체 찾기
        max_iou = 0
        assigned_class = "Unknown"
        conf_value = 0.0
        color = (0, 255, 0)

        for mapping in class_mappings:
            iou = compute_iou((x1, y1, x2, y2), mapping['bbox'])
            if iou > max_iou:
                max_iou = iou
                assigned_class = mapping['class']
                conf_value = mapping['conf']
                color = class_colors[assigned_class]

        # 임계값 이상일 경우 클래스 할당
        if max_iou > 0.1:
            label = f"{assigned_class}: {conf_value:.2f}%"
        else:
            label = "Unknown"

        cv2.rectangle(im0, (x1, y1), (x2, y2), color, 4)
        cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # 클래스별 카운팅 결과 텍스트 추가
    count_text = " | ".join([f"{cls}: {count}" for cls, count in class_count.items()])
    im0 = add_transparent_textbox(im0, count_text, (10, 60), font_scale=2.0, font_thickness=3)

    # 감지된 객체 수 출력
    print(f"Frame {frame_count}: {class_count}")

    video_writer.write(im0)
    frame_count += 1

# 리소스 해제
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# 동영상 저장 경로 출력
print(f"Detection results saved to {output_video_path}")






