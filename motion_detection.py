import os
import cv2

VIDEO = "videos/people-detection.mp4"   # 0 = webcam
SAVE_DIR = "motion_frames"            # thư mục lưu ảnh có chuyển động
SHOW = True                           # False = chạy không mở cửa sổ

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Không mở được video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30    # webcam có thể trả về 0 -> mặc định 30
delay = int(1000 / fps)                  # ms chờ mỗi frame để phát đúng tốc độ thật
save_every = int(fps)                    # khi có chuyển động: lưu 1 ảnh mỗi giây
os.makedirs(SAVE_DIR, exist_ok=True)

frames = []                   # lưu các frame grayscale gần nhất
gap = 5                       # so với frame cách 5 frame trước
count = 0                     # đếm số frame đã xử lý
saved = 0                     # đếm số ảnh đã lưu
MIN_AREA = 1000               # ngưỡng diện tích (độ nhạy)

while True:
    ret, frame = cap.read()   # ret: True/False (đọc được không), frame: ảnh
    if not ret:               # không còn frame (hết video / mất camera)
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)   # giảm nhiễu trước khi so sánh
    frames.append(gray)

    if len(frames) > gap + 1:     # giữ tối đa 6 frame
        frames.pop(0)             # bỏ frame cũ nhất

    cv2.putText(frame, f"Frame: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if len(frames) > gap:         # đủ 6 frame mới bắt đầu so sánh
        diff = cv2.absdiff(frames[0], frames[-1])
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)   # nối các mảnh rời của cùng một vật
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        big = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
        for c in big:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if big:                   # có ít nhất một vùng chuyển động đủ lớn
            cv2.putText(frame, "Motion Detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if count % save_every == 0:
                path = f"{SAVE_DIR}/motion_frame_{count}.jpg"
                cv2.imwrite(path, frame)
                saved += 1
                print(f"Saved {path}")

    count += 1
    if SHOW:
        cv2.imshow("Motion Detection", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):     # nhấn q để thoát
            break

cap.release()                 # giải phóng camera / file video
cv2.destroyAllWindows()
print(f"Xong: {count} frame, lưu {saved} ảnh vào {SAVE_DIR}/")
