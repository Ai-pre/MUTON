import cv2
import requests

URL = "http://127.0.0.1:5000/upload_frame"  # 먼저 같은 PC에서 테스트

cap = cv2.VideoCapture(0)  # 노트북 웹캠

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # JPEG로 인코딩
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        continue

    files = {
        "frame": ("frame.jpg", buf.tobytes(), "image/jpeg")
    }

    res = requests.post(URL, files=files)
    print(res.status_code, res.json())

    # ESC 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
