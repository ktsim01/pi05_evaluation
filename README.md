# pi05 evaluation

This repository contains the evaluation setup for pi05 with robot gripper, camera streaming, and inference scripts.

## Important note about `openpi`

You need to clone and use your own `openpi` repository for this evaluation (do not rely on someone else's `openpi` fork/credentials).

## Quick start

### 1) Bring up the Robotiq gripper

From the repository root:

```bash
cd robotiq_deoxys_control/deoxys
python3 auto_scripts/robotiq_gripper.py --comport /dev/ttyUSB1
```

### 2) Start USB cameras

From the repository root:

```bash
cd real_world_visual_planning
python3 usb_camera_to_redis.py
```

### 3) Run inference

From the repository root:

```bash
cd real_world_visual_planning
python3 panda_log.py --external_camera left
```

