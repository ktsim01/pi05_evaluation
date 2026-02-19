import cv2
import redis
import json
import struct
import time
import numpy as np

# =========================
# Configuration
# =========================

CAMERA_MAP = {
    0: '/dev/video0',  # external mono camera
    1: '/dev/video4',  # wrist stereo camera
}

CAMERA_NAMES = {
    0: "camera_rs_0",
    1: "camera_rs_1",
}

WRIST_CAM_ID = 1
USE_LEFT_EYE = True  # False = right eye

FPS = 30
MAX_FAILED_READS = 1
REOPEN_DELAY_SEC = 1.0

# =========================
# Redis
# =========================

r_info = redis.StrictRedis(
    host='127.0.0.1',
    port=6379,
    db=0,
    charset='utf-8',
    decode_responses=True
)

r_img = redis.StrictRedis(
    host='127.0.0.1',
    port=6379,
    db=0,
    charset='utf-8',
    decode_responses=False
)

# =========================
# Camera helpers
# =========================

def open_camera(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    # Reduce latency & buffering issues
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def reopen_camera(cam_id):
    device = CAMERA_MAP[cam_id]
    print(f"[INFO] Reopening camera {device}")

    try:
        caps[cam_id].release()
    except Exception:
        pass

    time.sleep(REOPEN_DELAY_SEC)

    cap = open_camera(device)
    if cap is None:
        print(f"[ERROR] Failed to reopen {device}")
        return False

    caps[cam_id] = cap
    failed_reads[cam_id] = 0
    print(f"[INFO] Successfully reopened {device}")
    return True


def publish_frame(cam_id, frame):
    h, w, c = frame.shape

    img_info = {
        "height": h,
        "width": w,
        "channels": c,
        "camera_type": "rs",
    }

    r_info.set(
        f"{CAMERA_NAMES[cam_id]}::last_img_info",
        json.dumps(img_info)
    )

    shape = struct.pack(">III", h, w, c)
    r_img.set(
        f"{CAMERA_NAMES[cam_id]}::last_img_color",
        shape + frame.tobytes()
    )

# =========================
# Open cameras
# =========================

caps = {}
failed_reads = {}

for cam_id, device in CAMERA_MAP.items():
    cap = open_camera(device)
    if cap is None:
        raise RuntimeError(f"Cannot open camera {device}")

    caps[cam_id] = cap
    failed_reads[cam_id] = 0
    print(f"Opened camera {device} as {CAMERA_NAMES[cam_id]}")

# =========================
# Main loop
# =========================

try:
    print("Streaming cameras to Redis. Press Ctrl+C to stop.")

    while True:
        for cam_id, cap in caps.items():
            ret, frame = cap.read()

            if not ret:
                failed_reads[cam_id] += 1
                print(
                    f"[WARN] Failed read {failed_reads[cam_id]} "
                    f"from {CAMERA_MAP[cam_id]}"
                )

                if failed_reads[cam_id] >= MAX_FAILED_READS:
                    reopen_camera(cam_id)

                continue

            failed_reads[cam_id] = 0

            # =========================
            # Stereo wrist camera split
            # =========================
            if cam_id == WRIST_CAM_ID:
                h, w, _ = frame.shape
                half_w = w // 2
                frame = frame[:, :half_w] if USE_LEFT_EYE else frame[:, half_w:]

            # OpenCV → RGB (panda_log expects RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            publish_frame(cam_id, frame)

        time.sleep(1 / FPS)

except KeyboardInterrupt:
    print("Stopping camera stream...")

finally:
    for cap in caps.values():
        cap.release()
