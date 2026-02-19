"""Teleoperating robot arm with a OculusQuest to collect demonstration data"""

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from deoxys_vision.networking.camera_redis_interface import \
    CameraRedisSubInterface
from easydict import EasyDict

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
# from deoxys.k4a_interface import K4aInterface
from deoxys.utils.io_devices import OculusQuest
from deoxys.utils.log_utils import get_deoxys_example_logger
import h5py
from deoxys.experimental.motion_utils import follow_joint_traj, reset_joints_to

logger = get_deoxys_example_logger()

import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor_id",
        type=int,
        default=9583,
    )
    parser.add_argument(
        "--product_id",
        type=int,
        default=50741,
    )
    robot_config_parse_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # Setting up robot, teleop, and camera interfaces
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg))
    device = OculusQuest()
    device.start_control()
    breakpoint()
    camera_ids = []
    camera_types = []
    camera_info = EasyDict({
        "camera_id": camera_ids,
        "camera_type": camera_types,
        "camera_name": [f"camera_{camera_types[id]}_{camera_ids[id]}" for id,_ in enumerate(camera_ids)]
    })
    cr_interfaces = {}
    for camera_id in camera_ids:
        print(camera_id, 'id')
        if camera_id == 2:
            cr_interface = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id, use_color=False, use_thermal=True)
            print(cr_interface.use_color, cr_interface.use_thermal)
        else:
            cr_interface = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id)

        cr_interface.start()
        cr_interfaces[camera_id] = cr_interface

    print(args.controller_cfg)
    controller_cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()

    # demo_file = h5py.File(demo_file_name, "w")
    controller_type = args.controller_type
    print(controller_type)
    
    args.folder.mkdir(parents=True, exist_ok=True)
    experiment_id = 0
    logger.info(f"Saving to {args.folder}")
    
    init_joints = np.array([0.029808449369959012, 0.3005430567421663, -0.0019222913376851023, -2.153063709726163, -0.07925823536534997, 2.531140358617818, 0.8007571734331206])
    
    while True:
        data = {"action": [], "ee_states": [], "joint_states": [], "gripper_states": []}
        for camera_id in camera_ids:
            data[f"camera_{camera_id}"] = []
        
        i = 0
        start = False

        previous_state_dict = None
        reset_joints_to(robot_interface, init_joints)
        print(robot_interface.last_eef_pose)
        print(robot_interface.last_eef_pose[2,3])
        init_pos = robot_interface.last_eef_pose[2,3]
        time.sleep(1)

        print('Beginning Trajectory')
        while True:
            start_time = time.time_ns()            
            robot_pos = robot_interface.last_eef_pose
            robot_gripper = robot_interface.last_gripper_q

            state_dict = {"cartesian_position": robot_pos, "gripper_position": robot_gripper}
            action, log, record = device.get_action(state_dict)
            if record:
                i += 1

            # set unused orientation dims to 0
            if controller_type == "OSC_YAW":
                action[3:5] = 0.0
            elif controller_type == "OSC_POSITION":
                action[3:6] = 0.0
            
            #action[2:5] = 0 #null_ac[2:5]
            #print(robot_interface.last_eef_pose[2,3])
            #action[2] = (init_pos - robot_interface.last_eef_pose[2,3])


            
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )

            if len(robot_interface._state_buffer) == 0:
                continue
            last_state = robot_interface._state_buffer[-1]
            last_gripper_state = robot_interface._gripper_state_buffer[-1]
            if np.linalg.norm(action[:-1]) < 1e-3 and not start:
                continue

            start = True
            # print(action.shape)
            # Record ee pose,  joints

            if record:
                data["action"].append(action)

                state_dict = {
                    "ee_states": np.array(last_state.O_T_EE),
                    "joint_states": np.array(last_state.q),
                    "gripper_states": np.array(last_gripper_state.width),
                }
                print('joint', last_state.q)

                if previous_state_dict is not None:
                    for proprio_key in state_dict.keys():
                        proprio_state = state_dict[proprio_key]
                        if np.sum(np.abs(proprio_state)) <= 1e-6:
                            proprio_state = previous_state_dict[proprio_key]
                        state_dict[proprio_key] = np.copy(proprio_state)
                for proprio_key in state_dict.keys():
                    data[proprio_key].append(state_dict[proprio_key])

                previous_state_dict = state_dict
            
                for camera_id in camera_ids:
                    #print(camera_id)
                    #print(cr_interfaces[camera_id])
                    #print(cr_interfaces[camera_id].use_color)
                    #print(cr_interfaces[camera_id].use_thermal)
                    img = cr_interfaces[camera_id].get_img()
                    if camera_id == 0:
                        img  = cv2.resize(img['color'], (256, 256), interpolation=cv2.INTER_LINEAR)
                    if camera_id == 1:
                        h, w, c = img['color'].shape
                        start_x = (w - 690) // 2  # (1280 - 720) // 2 = 280
                        img = img['color'][:, start_x:start_x + 700]
                        img = img[50:615]
                        img  = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    if camera_id == 2:
                        img  = cv2.resize(img['thermal'], (256, 256), interpolation=cv2.INTER_LINEAR)

                    data[f"camera_{camera_id}"].append(img)

            end_time = time.time_ns()

            if log is not None:
                break
        
        robot_interface.close()
        if log == "OPTIMAL":
            folder = "/data/" + str(args.folder) + "/optimal"
        if log == "SUBOPT":
            folder = "/data/" + str(args.folder) + "/subopt"
        
        print(f"saving to {folder}")
        os.makedirs(folder, exist_ok=True)

        # Find the next available trajectory file name
        experiment_id = 0
        for path in Path(folder).glob("traj_*.hdf5"):
            try:
                file_id = int(path.stem.split("_")[-1])
                experiment_id = max(experiment_id, file_id + 1)
            except ValueError:
                pass

        file_path = os.path.join(folder, f"traj_{experiment_id:04d}.hdf5")
        with h5py.File(file_path, "w") as h5py_file:
            config_dict = {
                "controller_cfg": dict(controller_cfg),
                "controller_type": controller_type,
            }
            grp = h5py_file.create_group("data")
            grp.attrs["config"] = json.dumps(config_dict)

            grp.create_dataset("actions", data=np.array(data["action"]))
            grp.create_dataset("ee_states", data=np.array(data["ee_states"]))
            grp.create_dataset("joint_states", data=np.array(data["joint_states"]))
            grp.create_dataset("gripper_states", data=np.array(data["gripper_states"]))

            for camera_id in camera_ids:
                grp.create_dataset(f"camera_{camera_id}", data=np.array(data[f"camera_{camera_id}"]))
                cr_interfaces[camera_id].stop()

        traj_len = len(data["action"])
        print(f"Trajectory {experiment_id} length {traj_len}")
    #robot_interface.close()

    
if __name__ == "__main__":
    main()
