import cv2

for idx in range(3):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    print(idx, "opened:", cap.isOpened())
    cap.release()
