# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from typing import Optional
# from droid.robot_env import RobotEnv
import tqdm
import tyro
from deoxys_vision.networking.camera_redis_interface import \
    CameraRedisSubInterface
from easydict import EasyDict
import imageio
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
# from deoxys.k4a_interface import K4aInterface
# from deoxys.utils.io_devices import OculusQuest
from deoxys.utils.log_utils import get_deoxys_example_logger
import h5py
from deoxys.experimental.motion_utils import follow_joint_traj, reset_joints_to
import cv2
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
faulthandler.enable()
OPEN = 1
CLOSED = -1


## some customized function to deal with joint_velocity to joint_delta 
def joint_velocity_to_delta(joint_velocity):
    # import pdb; pdb.set_trace()
    relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    max_joint_delta = relative_max_joint_delta.max()

    if isinstance(joint_velocity, list):
        joint_velocity = np.array(joint_velocity)

    relative_max_joint_vel = joint_delta_to_velocity(relative_max_joint_delta)
    max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()

    if max_joint_vel_norm > 1:
        joint_velocity = joint_velocity / max_joint_vel_norm

    joint_delta = joint_velocity * max_joint_delta

    return joint_delta

def joint_delta_to_velocity(joint_delta):
    relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    max_joint_delta = relative_max_joint_delta.max()
    if isinstance(joint_delta, list):
        joint_delta = np.array(joint_delta)

    return joint_delta / max_joint_delta

@dataclasses.dataclass
class Args:
    # Hardware parameters
    # left_camera_id: str = "<your_camera_id>"  # e.g., "24259877"
    # right_camera_id: str = "<your_camera_id>"  # e.g., "24514023"
    # wrist_camera_id: str = "<your_camera_id>"  # e.g., "13062452"
    interface_cfg: str = "configs/charmander.yml"
    
    folder: Path = Path('/home/ksaha/kyutae/real_world_visual_planning/logs')

    controller_type: str="JOINT_IMPEDANCE"
    ## they are using joint velocity control not traditional ee pose control 
    controller_cfg: str="configs/joint-impedance-controller.yml"

    # Policy parameters
    # external_camera: str | None = (
    #     None  # which external camera should be fed to the policy, choose from ["left", "right"]
    # )

    external_camera: Optional[str] = None

    # Rollout parameters
    max_timesteps: int = 1000 #1000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    policy: str = "DROID" # the pi05 inferenece checkpoint



# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt

def get_imgs_from_env(cr_interfaces, camera_ids):
    data = {}
    
    for camera_id in camera_ids:
        #print(camera_id)
        #print(cr_interfaces[camera_id])
        #print(cr_interfaces[camera_id].use_color)
        #print(cr_interfaces[camera_id].use_thermal)
        img = cr_interfaces[camera_id].get_img()
        # third person
        if camera_id == 0:
            h, w, c = img['color'].shape
            # imageio.imwrite("uncropped_wrist_0.png", img['color'])
            # breakpoint()
            # start_x = (w - 690) // 2  # (1280 - 720) // 2 = 280
            # img = img['color'][:, start_x:start_x + 700]
            # img = img[50:615]
            # img  = cv2.resize(img['color'], (224, 224), interpolation=cv2.INTER_LINEAR)
            img = img['color']
        if camera_id == 1:
            h, w, c = img['color'].shape
            # start_x = (w - 690) // 2  # (1280 - 720) // 2 = 280
            # img = img['color'][:, start_x:start_x + 700]
            # img = img[50:615]
            # img  = cv2.resize(img['color'], (224, 224), interpolation=cv2.INTER_LINEAR)
            img = img['color']
        if camera_id == 2:
            img = img['color']
        # if camera_id == 2:
        #     img  = cv2.resize(img['thermal'], (256, 256), interpolation=cv2.INTER_LINEAR)
        ## convert bgr to rgb 
        # breakpoint()
        # import pdb; pdb.set_trace()
        data[f"camera_{camera_id}"] = img
    return data

def get_robot_joints(robot_interface) -> np.ndarray:
    while True:
        print("Trying to get robot state...")
        if len(robot_interface._gripper_state_buffer) > 0 and len(robot_interface._state_buffer) > 0:
            last_gripper_state = robot_interface._gripper_state_buffer[-1] ### NOTE: check if this gripper state is actually normalized
            gripper_state = last_gripper_state.width 
            
            last_state = robot_interface._state_buffer[-1]
            ee_pos = last_state.O_T_EE 
            joint_state = last_state.q

            return last_state, last_gripper_state, ee_pos, gripper_state, joint_state


def step(action, robot_interface,controller_cfg, controller_type = 'OSC_YAW', step_count=0 ):
    # set unused orientation dims to 0
    # if controller_type == "OSC_YAW":
    #     action[3:5] = 0.0
    # elif controller_type == "OSC_POSITION":
    #     action[3:6] = 0.0
    ## action space is 7d joint velocity + 1d gripper position
    # breakpoint()

    if controller_type == "JOINT_POSITION" or controller_type == 'JOINT_IMPEDANCE':
        if args.policy == "DROID":
            # pi05-DROID return the joint velocity as the action space
            joint_velocity = action[:7]
            delta = joint_velocity_to_delta(joint_velocity) # 2nd exp: vel -> pos
            joint_position = robot_interface.last_q 
            robot_action = joint_position + delta 
        elif args.policy == "FRANKA":
            # pi05-BASE return the joint angles in radiant as the action space
            print("raw action size ", len(action))
            joint_position = action[:7]
            robot_action = joint_position
            # breakpoint()

        gripper_pos = action[-1]
        print("the comamnded gripper position is: ", gripper_pos)
        # if step_count == 0:
        #     ## run 10 steps and try to reach that joint position
        #     num_intermediate_arrays = 8

        #     # Create weights for linear interpolation
        #     # The weights will range from 0 to 1 in 'num_intermediate_arrays + 1' steps,
        #     # excluding 0 and 1 themselves for the *intermediate* arrays.
        #     weights = np.linspace(0, 1, num_intermediate_arrays + 2)[1:-1]

        #     interpolated_arrays = []
        #     for w in weights:
        #         # Linear interpolation formula: (1 - w) * a1 + w * a2
        #         interpolated_array = (1 - w) * joint_position + w * robot_action[:7]
        #         interpolated_arrays.append(interpolated_array)
        #             # joint_action = joint_position + delta 
        #             # print('delta', joint_action)
        #     for middle_step in interpolated_arrays:
        #         robot_action = middle_step.tolist() + action[-1:].tolist()
        #         robot_interface.control(controller_type = controller_type, action = robot_action, controller_cfg=controller_cfg)
        print('current joint position', joint_position)
        # print('current joint velocity', joint_velocity)
        # robot_action = action # 1st exp: raw vel actions
        robot_action = robot_action.tolist() + [gripper_pos] ### DONE: check if the gripper action is flipped inside the controller
        ### this is what official driod dataest do
        

        # robot_action = joint_action.tolist() + action[-1:].tolist()
        # print('current joint action', robot_action)
        assert len(robot_action)==8
    else: 
        raise NotImplementedError
    
    
    #action[2:5] = 0 #null_ac[2:5]
    #print(robot_interface.last_eef_pose[2,3])
    #action[2] = (init_pos - robot_interface.last_eef_pose[2,3])


    # breakpoint()
    robot_interface.control(
        controller_type=controller_type,
        action=robot_action,
        controller_cfg=controller_cfg,
    )
    return 
     


def get_observation(cr_interfaces, robot_interface, camera_ids):
    camera_data = get_imgs_from_env(cr_interfaces, camera_ids) ### NOTE: check if the camera is rgb
    last_state, last_gripper_state, ee_pos, gripper_state, joint_state = get_robot_joints(robot_interface)
    obs = {}
    obs['joint_position'] = np.array(joint_state) 
    obs['gripper_position'] = np.array(gripper_state) ### NOTE: shouldn't this be normalized somewhere?
    obs['gripper_position'] = 1 - obs['gripper_position'] / 0.08
    obs['gripper_position'] = np.clip(obs['gripper_position'], 0, 1)
    obs['eef_pos'] = np.array(ee_pos)
    obs['wrist_image'] = camera_data['camera_1'] ## assume 0 is for zed
    # camera_0 is the external camera, can be used as either left or right
    obs['left_image'] = camera_data['camera_0']
    obs['right_image'] = camera_data['camera_0'] ### NOTE: why are we passing the same image as both left and right?
    # import pdb; pdb.set_trace()
    return obs 

def main(args: Args):
    # DROID data collection frequency -- we slow down execution to match this frequency
    DROID_CONTROL_FREQUENCY = 15 if args.policy=="DROID" else 50
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    # robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg), control_freq=DROID_CONTROL_FREQUENCY)
    robot_interface = FrankaInterface(
    "configs/charmander.yml", 
    control_freq=DROID_CONTROL_FREQUENCY
    )

    camera_ids = [0,1]#, 2] # 0 is 3rd person, 1 is wrist
    # camera_types = ['zed', 'zed']# 'zed']
    camera_types = ['rs', 'rs']
    # Wait a moment to ensure camera data is available in Redis
    import redis
    r_check = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)
    print("Waiting for camera data to be available in Redis...")
    for camera_id in camera_ids:
        camera_name = f"camera_{camera_types[camera_ids.index(camera_id)]}_{camera_id}"
        max_wait = 5.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if r_check.exists(f"{camera_name}::last_img_info") and r_check.exists(f"{camera_name}::last_img_color"):
                print(f"Found data for {camera_name}")
                break
            time.sleep(0.1)
        else:
            print(f"Warning: No data found for {camera_name} after {max_wait}s")
    
    cr_interfaces = {}
    for idx, camera_id in enumerate(camera_ids):
        print(camera_id, 'id')
        # Create camera_info - camera_name must be a STRING, not a list!
        # The interface uses camera_name directly in f-strings to construct Redis keys
        camera_name = f"camera_{camera_types[idx]}_{camera_id}"
        camera_info = EasyDict({
            "camera_id": camera_id,  # Can be int or list, but camera_name must be string
            "camera_type": camera_types[idx],
            "camera_name": camera_name  # STRING, not list!
        })
        print(f"Creating interface for camera: {camera_name}")
        
        # Verify Redis keys exist before starting
        info_key = f"{camera_name}::last_img_info"
        color_key = f"{camera_name}::last_img_color"
        if not r_check.exists(info_key):
            print(f"ERROR: Redis key {info_key} does not exist!")
        if not r_check.exists(color_key):
            print(f"ERROR: Redis key {color_key} does not exist!")
        
        # if camera_id == 2:
        #     cr_interface = CameraRedisSubInterface(camera_info=camera_info, camera_id=camera_id, use_color=False, use_thermal=True)
        #     print(cr_interface.use_color, cr_interface.use_thermal)
        # else:
        try:
            cr_interface = CameraRedisSubInterface(camera_info=camera_info)
            print(f"Interface created, calling start()...")
            cr_interface.start()
            print(f"Interface started successfully for {camera_name}")
            cr_interfaces[camera_id] = cr_interface
        except Exception as e:
            print(f"ERROR starting interface for {camera_name}: {e}")
            import traceback
            traceback.print_exc()
            raise

    print(args.controller_cfg)
    # controller_cfg = YamlConfig(
    #     os.path.join(config_root, args.controller_cfg)
    # ).as_easydict()
    config_path = "configs"

    controller_cfg = YamlConfig(
    f"{config_path}/joint-impedance-controller.yml"
    ).as_easydict()

    # demo_file = h5py.File(demo_file_name, "w")
    controller_type = args.controller_type
    print(controller_type)
    
    args.folder.mkdir(parents=True, exist_ok=True)
    experiment_id = 0
    # logger.info(f"Saving to {args.folder}")
    print(f'Saving to this folder {args.folder}')
    print("Created the droid env!")
    # init_joints = np.array([ 0.094, -0.198 ,-0.025 ,-2.47 , -0.018 , 2.304,  0.849]) 
    # init_joints = np.array([ 0.0933692 ,  0.07232527 ,-0.03192432, -2.17384338 ,-0.01927867,  2.26411851,0.07160476]) # lower
    init_joints = np.array([-0.13063249, -0.4617808, -0.01840502, -2.16992572, -0.03334957, 1.6950341, -0.19681183]) # higher
    reset_joints_to(robot_interface, init_joints, gripper_open=True)

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    print(get_robot_joints(robot_interface))
    trial_num = 0
    today_date = datetime.datetime.now().strftime("%Y_%m_%d")
    while True:
        trial_num += 1
        instruction = input("Enter instruction: ")
        # instruction = "put the cup in the basket"

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        video_wrist = []
        observations = []
        executed_actions = []
        predicted_actions = []
        request_data_list = []  # Store request_data for debugging
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = get_observation(cr_interfaces=cr_interfaces,robot_interface= robot_interface, camera_ids=camera_ids)
               
                # Record observation
                observations.append({
                    'joint_position': curr_obs['joint_position'],
                    'gripper_position': curr_obs['gripper_position'],
                    'wrist_image': curr_obs['wrist_image'],
                    'left_image': curr_obs[f'{args.external_camera}_image'], ### NOTE: why are we using the left image here?
                    # 'right_image': curr_obs['right_image'],
                    'eef_pos': curr_obs['eef_pos']
                })



                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.

                    video.append(curr_obs[f"{args.external_camera}_image"])
                    video_wrist.append(curr_obs["wrist_image"])
                    
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        # "observation/exterior_image_1_right": image_tools.resize_with_pad(
                        #     curr_obs[f"{args.external_camera}_image"], 224, 224
                        # ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }
                    
                    # breakpoint()
                    # import matplotlib.pyplot as plt
                    # plt.imshow(request_data["observation/exterior_image_1_left"])
                    # plt.show()
                    # plt.imshow(request_data["observation/wrist_image_left"])
                    # plt.show()

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    predicted_actions.append(np.array(pred_action_chunk))
           
                    assert pred_action_chunk.shape == (15, 8) if args.policy=="DROID" else (50,8) #(10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed] 
                # action = action_sequences[t_step]
                actions_from_chunk_completed += 1
                
                # Binarize gripper action
                # if action[-1].item() > 0.5:
                if action[-1] > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], -np.ones((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

              
            
          
                # print(f"gripper action {action[-1]}")
              
                # Record executed action
                executed_actions.append(np.array(action))
                step(action, robot_interface=robot_interface,controller_cfg=controller_cfg, controller_type=controller_type, step_count=t_step)
                
                # if action[-1] == 1:
                #     time.sleep(1)
                #     reset_joints_to(robot_interface, init_joints, gripper_open=False)
                #     break
                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break
        # breakpoint()
        # video = np.stack(video)
        # video_wrist = np.stack(video_wrist)
        # save_filename = str(args.folder) + "/" + "video_" + timestamp


        success: str | float | None = None
        while not isinstance(success, float):
            success_input = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success_input.lower() == "y":
                success = 1.0
            elif success_input.lower() == "n":
                success = 0.0
            else:
                try:
                    success = float(success_input) / 100
                except ValueError:
                    print(f"Invalid input. Please enter 'y', 'n', or a number between 0-100.")
                    success = None
                    continue
            
            if success is not None and not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")
                success = None
        
        # Save images to folder structure: today's_date/trialN_success/wrist/ and today's_date/trialN_success/external_camera/
        success_str = "success" if success >= 0.5 else "failure"
        trial_folder_name = f"trial{trial_num}_{success_str}"
        base_image_folder = args.folder / today_date / trial_folder_name
        wrist_folder = base_image_folder / "wrist"
        external_camera_folder = base_image_folder / "external_camera"
        
        # Create directories
        wrist_folder.mkdir(parents=True, exist_ok=True)
        external_camera_folder.mkdir(parents=True, exist_ok=True)
        
        # Save wrist images
        print(f"Saving {len(video_wrist)} wrist images to {wrist_folder}")
        for idx, img in enumerate(video_wrist):
            # Images are already in RGB format from camera interface
            if isinstance(img, np.ndarray):
                image_path = wrist_folder / f"frame_{idx:05d}.png"
                Image.fromarray(img).save(image_path)
            else:
                # If it's already a PIL Image or other format
                image_path = wrist_folder / f"frame_{idx:05d}.png"
                Image.fromarray(np.array(img)).save(image_path)
        
        # Save external camera images
        print(f"Saving {len(video)} external camera images to {external_camera_folder}")
        for idx, img in enumerate(video):
            # Images are already in RGB format from camera interface
            if isinstance(img, np.ndarray):
                image_path = external_camera_folder / f"frame_{idx:05d}.png"
                Image.fromarray(img).save(image_path)
            else:
                # If it's already a PIL Image or other format
                image_path = external_camera_folder / f"frame_{idx:05d}.png"
                Image.fromarray(np.array(img)).save(image_path)
        
        # print(f"Saved images to {base_image_folder}")
        
        # ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")
        # ImageSequenceClip(list(video_wrist), fps=10).write_videofile(save_filename + "_wrist.mp4", codec="libx264")
        
        # save_resp = input("Save trajectory data to PKL? (y/n) ").lower().strip()
        save_resp='y'
        if save_resp == 'y':
            data = {
                'language_instruction': instruction,
                'observations': observations,
                'action_chunks': predicted_actions,
           
                'executed_actions': executed_actions,
                'timestamp': timestamp,
                'open-loop-horizon': args.open_loop_horizon,
                'policy': args.policy,
                'external_camera': args.external_camera,
                'success': success
            }

            # Save trajectory in the trial folder
            out_path = base_image_folder / f"trajectory_{timestamp}.pkl"
            with open(out_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved trajectory to {out_path}")
        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
    
        reset_joints_to(robot_interface, init_joints, gripper_open=True)
        

    # os.makedirs("results", exist_ok=True)
    # timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    # csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    # df.to_csv(csv_filename)
    # print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
