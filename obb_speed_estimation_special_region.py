import cv2
from ultralytics import YOLO, solutions
import numpy as np
import os

# YOLO 모델 로드
model = YOLO(r"D:\yolo11l-obb.pt")  # 기존 YOLO 모델 경로
names = {9: "large vehicle", 10: "small vehicle"}

# 비디오 파일 설정
input_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\DJI_0076.MP4"
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "비디오 파일을 읽는 중 오류 발생"
w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

# 결과 비디오 저장 설정
output_video_path = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\obb_detection_with_speed_count.avi"
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 50프레임마다 캡처한 프레임을 저장할 폴더
capture_folder = "C:\\Users\\dromii\\20240401-송도\\M3C\\200m\\capture_frames3"
os.makedirs(capture_folder, exist_ok=True)

# SpeedEstimator 설정
speed_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
speed_estimator = solutions.SpeedEstimator(model="yolo11n.pt", region=speed_region, show=True)

# 색상 설정
colors = {"small vehicle": (255, 0, 0), "large vehicle": (0, 0, 255)}  # 파란색과 빨간색
speed_color = (0, 255, 0)  # 속도 색상 초록색

# 타일 기반 예측 및 표시 함수
def process_and_annotate_frame(frame):
    results = model(frame)
    small_vehicle_count = 0
    large_vehicle_count = 0

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

            # 차량 카운트
            if class_id == 9:
                large_vehicle_count += 1
            elif class_id == 10:
                small_vehicle_count += 1

            # SpeedEstimator를 통한 속도 추정 수행
            out_frame = speed_estimator.estimate_speed(frame)

            # OBB 시각화 및 클래스명과 정확도 표시
            pts = obb_coords.astype(int)
            cv2.polylines(out_frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(out_frame, f"{class_name} {confidence:.2f}", (int(cx), int(cy) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 우측 상단에 차량 개수 카운팅 표시
    overlay = out_frame.copy()
    cv2.rectangle(overlay, (w - 200, 10), (w - 10, 80), (50, 50, 50), -1)  # 회색 반투명 배경 박스
    alpha = 0.6
    out_frame = cv2.addWeighted(overlay, alpha, out_frame, 1 - alpha, 0)
    cv2.putText(out_frame, f"Small Vehicles: {small_vehicle_count}", (w - 190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out_frame, f"Large Vehicles: {large_vehicle_count}", (w - 190, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return out_frame

# 프레임별로 비디오 처리
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("비디오 프레임이 없거나 처리가 완료되었습니다.")
        break

    # 프레임에 속도 추정 및 시각화 적용
    annotated_frame = process_and_annotate_frame(frame)

    # 결과 비디오 저장
    video_writer.write(annotated_frame)

    # 50 프레임마다 캡처하여 이미지 저장
    if frame_count % 50 == 0:
        capture_path = os.path.join(capture_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(capture_path, annotated_frame)

    # 결과 프레임을 화면에 표시
    cv2.imshow("Vehicle Detection with Speed Estimation", annotated_frame)
    frame_count += 1

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업 완료 후 자원 해제
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Detection results saved to {output_video_path}")
print(f"Captured frames saved to {capture_folder}")

