# check_camera_warmup.py
import cv2

for idx in [0, 1]:
    cap = cv2.VideoCapture(idx)

    if not cap.isOpened():
        print(f"Camera {idx}: failed to open")
        continue

    # Set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm up by grabbing a bunch of frames
    for i in range(30):
        ok, frame = cap.read()

    ok, frame = cap.read()
    cap.release()

    print(f"Camera {idx}: read_ok={ok}")
    if ok:
        cv2.imwrite(f"cam{idx}_warm.png", frame)
        print(f"Saved cam{idx}_warm.png")

print("Done.")
