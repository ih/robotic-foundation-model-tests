"""Object detection via background subtraction.

Compares a current camera frame against a reference 'empty desk' image
to find objects that have been placed on the desk.
"""

import cv2
import numpy as np


def detect_object(frame: np.ndarray, reference: np.ndarray,
                  threshold: int = 30, min_area: int = 500,
                  debug: bool = False) -> tuple | None:
    """Detect an object by comparing frame to a reference empty-desk image.

    Args:
        frame: Current camera frame (BGR).
        reference: Reference empty-desk image (BGR), same resolution as frame.
        threshold: Pixel intensity difference threshold for foreground detection.
        min_area: Minimum contour area in pixels to count as an object.
        debug: If True, show intermediate detection images.

    Returns:
        (centroid_x, centroid_y, bounding_box) if object found, else None.
        bounding_box is (x, y, w, h).
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
    gray_ref = cv2.GaussianBlur(gray_ref, (7, 7), 0)

    diff = cv2.absdiff(gray_frame, gray_ref)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_frame = frame.copy()
        cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 1)

    # Find largest contour above minimum area
    best = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area > best_area:
            best = contour
            best_area = area

    if best is None:
        if debug:
            cv2.imshow("Detection - Mask", mask)
            cv2.imshow("Detection - Frame", frame)
            cv2.waitKey(1)
        return None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    bbox = cv2.boundingRect(best)

    if debug:
        cv2.rectangle(debug_frame, (bbox[0], bbox[1]),
                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow("Detection - Mask", mask)
        cv2.imshow("Detection - Result", debug_frame)
        cv2.waitKey(1)

    return (cx, cy, bbox)


def capture_frame(camera_index: int) -> np.ndarray | None:
    """Capture a single frame from a camera.

    Opens the camera, grabs a frame, and closes it immediately.
    Uses DSHOW backend on Windows for compatibility.
    """
    import platform

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None

    # Warm up - discard first few frames
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame from camera {camera_index}")
        return None

    return frame


def capture_frame_from_open_cap(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Capture a frame from an already-open VideoCapture."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame
