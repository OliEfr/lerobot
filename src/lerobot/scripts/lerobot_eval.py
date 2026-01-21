#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import os
import shutil
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict
import cv2
import zmq
import msgpack

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.policies.diffusion.modeling_diffusion import RND
from lerobot.utils.constants import VISUAL_SERVOING_SETTINGS
import lerobot.policies.system2 as system2_module
from robosuite.utils.transform_utils import axisangle2quat, get_orientation_error

from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    bilinear_interpolate
)

SAVE_FRAMES = True

def transform_from_pixels_to_world(pixels, depth_map, camera_to_world_transform):
    pixels = pixels.astype(float)
    z = bilinear_interpolate(im=depth_map, x=pixels[..., 1], y=pixels[..., 0])
    z = z.reshape(-1, 1)  # shape [..., 1]

    # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
    # (note that we need to swap the first 2 dimensions of pixels to go from pixel indices
    # to camera coordinates)
    cam_pts = [pixels[..., 1:2] * z, pixels[..., 0:1] * z, z, np.ones_like(z)]
    cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do camera to robot frame transform
    mat_reshape = [1] * len(cam_pts.shape[:-1]) + [4, 4]
    cam_trans = camera_to_world_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    points = np.matmul(cam_trans, cam_pts[..., None])[..., 0]  # shape [..., 4]
    return points[..., :3]


class SAM3StreamClient:
    """Client for streaming frames to SAM3 and receiving segmented frames via ZMQ."""

    def __init__(
        self,
        send_endpoint: str = "tcp://localhost:5555",
        recv_endpoint: str = "tcp://localhost:5556",
        target_size: tuple[int, int] = (256, 256),  # (width, height) expected by SAM3
        original_size: tuple[int, int] = (256,256),
        output_dir: str | None = "output_sam3",  # Directory to save segmented frames
    ):
        """
        Initialize ZMQ sockets for SAM3 streaming.

        Args:
            send_endpoint: ZMQ endpoint to send frames to SAM3.
            recv_endpoint: ZMQ endpoint to receive segmented frames from SAM3.
            image_keys: Keys to look for in observation dict to find RGB images.
            target_size: (width, height) to resize frames to before sending to SAM3.
            output_dir: Directory to save segmented frames. None to disable saving.
        """
        self.target_size = target_size  # (width, height)
        self.original_size = original_size
        self.camera_names = ["1st_person", "3rd_person"]
        self.latest_segmented_frame: dict[str, np.ndarray] | None = None
        self._last_frame_shape: tuple[int, ...] | None = None
        self._frame_counter = 0
        self._output_dir = output_dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Initialize ZMQ context and sockets
        self._context = zmq.Context()

        # PUSH socket to send frames to SAM3
        self._sender = self._context.socket(zmq.PUSH)
        self._sender.setsockopt(zmq.SNDHWM, 1)  # Keep send queue small
        self._sender.setsockopt(zmq.LINGER, 0)  # Discard unsent messages on close
        self._sender.connect(send_endpoint)

        # SUB socket to receive segmented frames from SAM3
        self._receiver = self._context.socket(zmq.SUB)
        self._receiver.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
        self._receiver.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        self._receiver.setsockopt(zmq.RCVTIMEO, 0)  # Non-blocking receive
        self._receiver.connect(recv_endpoint)

    def _extract_rgb_frame(self, observation: dict) -> np.ndarray | None:
        """Extract RGB frame from observation dict.

        Handles PyTorch tensors in (B, C, H, W) format and converts to (H, W, C) uint8.
        """
        image_batch = np.concatenate(
            [
                observation[VISUAL_SERVOING_SETTINGS[cam]["lerobot_camera_name"]].cpu().numpy()
                for cam in self.camera_names
            ],
            axis=0,
        )

        image_batch = np.transpose(image_batch, (0, 2, 3, 1))

        # Convert float [0,1] to uint8 [0,255]
        if image_batch.dtype in (np.float32, np.float64):
            image_batch = (image_batch * 255).clip(0, 255).astype(np.uint8)

        return image_batch

    def send_frame_batched(
        self, observation: dict, sam3_stage: int, prompt: str | dict[str, str] | None = None
    ) -> bool:
        """
        Send batched RGB frames from observation to SAM3.

        Args:
            observation: Observation dict containing RGB images.
            sam3_stage: SAM3 stage counter. Server resets when this increases.
            prompt: Text prompt(s) for segmentation. Can be:
                - str: same prompt for all cameras (backward compatible)
                - dict: {camera_name: prompt} for per-camera prompts
                - None: uses empty string

        Returns:
            True if frames were sent, False otherwise.
        """
        rgb_batch = self._extract_rgb_frame(observation)
        if rgb_batch is None:
            return False

        # Store original shape for potential use
        self._last_frame_shape = rgb_batch.shape

        # Resize each frame in the batch to target size expected by SAM3
        # rgb_batch shape: (num_cameras, H, W, 3)
        target_w, target_h = self.target_size
        if rgb_batch.shape[1:3] != (target_h, target_w):
            resized_frames = []
            for i in range(rgb_batch.shape[0]):
                resized = cv2.resize(rgb_batch[i], (target_w, target_h))
                resized_frames.append(resized)
            rgb_batch = np.stack(resized_frames, axis=0)

        # Ensure uint8 and contiguous for tobytes()
        if rgb_batch.dtype != np.uint8:
            rgb_batch = rgb_batch.astype(np.uint8)
        rgb_batch = np.ascontiguousarray(rgb_batch)

        # Pack message with msgpack
        msg = msgpack.packb({
            "prompt": prompt,
            "sam3_stage": sam3_stage,
            "frames": rgb_batch.tobytes(),
        }, use_bin_type=True)

        try:
            self._sender.send(msg, zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False  # Skip if send would block
        

    def send_model_reset(self):
        """Send model reset signal to SAM3."""
        msg = msgpack.packb({
            "reset_model": True,
        }, use_bin_type=True)
        self._sender.send(msg)

    def drain_stale_messages(self) -> int:
        """Drain any stale messages from the receive buffer.

        This should be called at episode start to clear messages from previous episodes
        that may be buffered due to ZMQ CONFLATE setting.

        Returns:
            Number of messages drained.
        """
        count = 0
        while True:
            try:
                self._receiver.recv(zmq.NOBLOCK)
                count += 1
            except zmq.Again:
                break
        self.latest_segmented_frame = None
        return count

    def receive_segmented_frame(self) -> dict[str, np.ndarray] | None:
        """
        Receive latest batch of segmented frames from SAM3 (non-blocking).

        Returns:
            Dict mapping camera names to masks if available, None otherwise.
            Each mask is in original_size dimensions (H, W), uint8.
        """
        try:
            msg = self._receiver.recv(zmq.NOBLOCK)
            target_w, target_h = self.target_size
            num_cameras = len(self.camera_names)
            expected_pixels = target_h * target_w

            # Infer n_masks from message size: num_cameras * n_masks * H * W
            total_elements = len(msg)
            n_masks = total_elements // (num_cameras * expected_pixels)

            # Reshape to (num_cameras, n_masks, H, W)
            batch_masks = np.frombuffer(msg, dtype=np.uint8).reshape(
                (num_cameras, n_masks, target_h, target_w)
            )

            result = {}
            for i, cam_name in enumerate(self.camera_names):
                # Select first mask for this camera
                if batch_masks[i].shape[0] == 0:
                    continue
                mask = batch_masks[i, 0]  # Shape (H, W)

                # Resize to original size
                original_w, original_h = self.original_size
                mask = cv2.resize(mask, (original_w, original_h))
                result[cam_name] = mask

                # Save received masks for debugging
                if self._output_dir and SAVE_FRAMES:
                    path = os.path.join(self._output_dir, f"frame_{cam_name}_{self._frame_counter:05d}.png")
                    frame_to_save = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(path, frame_to_save)

            self._frame_counter += 1
            self.latest_segmented_frame = result if result else None
            return self.latest_segmented_frame

        except zmq.Again:
            pass  # No new segmented frame available
        return None

    def close(self):
        """Close ZMQ sockets and terminate context."""
        self._sender.close()
        self._receiver.close()
        self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

debug_dir = "output_centroid_debug"
if os.path.exists(debug_dir):
    shutil.rmtree(debug_dir)
os.makedirs(debug_dir)

def VS_first_person(
    segmented_frame: np.ndarray,
    depth_observation: np.ndarray,
    current_ee_pos: np.ndarray | None = None,
    camera_intrinsics: dict | None = None,
    gain: float = 0.1,
) -> np.ndarray:
    """
    Compute delta end-effector action to move towards the centroid of the segmented object.

    Args:
        segmented_frame: Binary segmentation mask from SAM3 (H, W, 1) or (H, W), uint8.
                         Non-zero pixels are the segmented object.
        depth_observation: Depth image (H, W) in meters.
        current_ee_pos: Current end-effector position [x, y, z]. If None, returns
                        target position instead of delta.
        camera_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy'. If None, uses defaults.
        gain: Proportional gain for the delta action (0-1).

    Returns:
        delta_action: np.ndarray of shape (6,) or (7,) containing [dx, dy, dz, 0, 0, 0, ...]
                      Rotations are set to zero.
    """

    # check shapes
    depth_observation = depth_observation[0]  # assuming (B,H,W,1)
    assert depth_observation.shape[:2] == segmented_frame.shape[:2]
    h, w = depth_observation.shape[:2]

    # Default camera intrinsics (adjust based on your camera)
    if camera_intrinsics is None:
        h, w = segmented_frame.shape[:2]
        camera_intrinsics = {
            "fx": w,  # focal length x (pixels)
            "fy": w,  # focal length y (pixels)
            "cx": w / 2,  # principal point x
            "cy": h / 2,  # principal point y
        }

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    # Find centroid of the segmented region
    moments = cv2.moments(segmented_frame)
    if moments["m00"] < 1e-6:
        # No segmented object found, return zero action
        return np.zeros(7, dtype=np.float32)

    centroid_x = int(round(moments["m10"] / moments["m00"]))  # x pixel coordinate
    centroid_y = int(round(moments["m01"] / moments["m00"]))  # y pixel coordinate



    # Sample depth in 5x5 neighborhood and take median (robust to noise)
    depth_patch = depth_observation[
        centroid_y - 2 : centroid_y + 3, centroid_x - 2 : centroid_x + 3
    ]
    valid_depths = depth_patch[depth_patch > 0]
    if len(valid_depths) == 0:
        # No valid depth, return zero action
        return np.zeros(7, dtype=np.float32)
    centroid_z = np.median(valid_depths)

    # Convert pixel coordinates + depth to 3D camera coordinates
    # Using pinhole camera model: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    target_x = (centroid_x - 256 // 2) / 640 # (centroid_x - cx) * centroid_z / fx
    target_y = (centroid_y - 256 // 2) / 640 # (centroid_y - cy) * centroid_z / fy
    target_z = centroid_z / 5

    target_pos = np.array([-target_y, target_x, -target_z], dtype=np.float32)

    # Return action with zero rotations: [dx, dy, dz, 0, 0, 0, gripper]
    action = np.zeros(7, dtype=np.float32)
    action[:3] = gain * target_pos
    
    # Debug: save segmented_frame with centroid overlay
    os.makedirs(debug_dir, exist_ok=True)
    debug_img = cv2.cvtColor(segmented_frame * 255, cv2.COLOR_GRAY2BGR)
    cv2.circle(debug_img, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)
    frame_idx = len(os.listdir(debug_dir))
    if SAVE_FRAMES:
        cv2.imwrite(
            os.path.join(debug_dir, f"segmented_frame_{frame_idx:05d}.png"), debug_img
        )

    return action

def move_to_home(
    observation: dict,
    gain: float = 0.5,
) -> np.ndarray:
    """
    Generate action to move the robot end-effector to a predefined home position.

    Args:
        observation: Observation dict containing robot state in observation["observation.state"]
        gain: Proportional gain for the delta action (0-1).

    Returns:
        action: np.ndarray of shape (7,) containing [dx, dy, dz, drx, dry, drz, gripper]
    """
    # Define home position (world frame)
    HOME_POS = np.array([-5.84646606e-02, 2.49015333e-12, 6.81279476e-01], dtype=np.float32)
    HOME_QUAT = np.array([9.99596605e-01, 2.46212834e-04, -2.84001205e-02, -6.99529629e-06], dtype=np.float32)

    # Extract current end-effector state from observation
    state = observation["observation.state"].cpu().numpy()[0]
    current_pos = state[:3]  # [x, y, z]
    current_aa = state[3:6]  # [ax, ay, az] axis-angle orientation

    # Compute position delta (simple subtraction)
    delta_pos = HOME_POS - current_pos

    # Compute rotation delta using proper quaternion math
    # Convert current axis-angle to quaternion
    q_current = axisangle2quat(current_aa)
    # Compute orientation error (returns 3D vector for impedance control)
    delta_rot = get_orientation_error(HOME_QUAT, q_current)

    # Apply gain to both deltas
    delta_pos = gain * delta_pos
    delta_rot = gain * delta_rot

    # Return action: [dx, dy, dz, drx, dry, drz, gripper]
    action = np.zeros(7, dtype=np.float32)
    action[:3] = delta_pos
    action[3:6] = delta_rot
    action[6] = 0  # gripper action (0 = no change)

    return action

def VS_third_person(
    env,
    segmented_frame: np.ndarray,
    depth_observation: np.ndarray,
    camera_info: dict,
    gain: np.ndarray,
    current_ee_pos: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute delta end-effector action to move towards the centroid of the segmented object.
    Uses 3rd person camera with proper camera-to-world transformation.

    Args:
        segmented_frame: Binary segmentation mask from SAM3 (H, W), uint8.
                         Non-zero pixels are the segmented object.
        depth_observation: Depth image (B, H, W) or (H, W) in meters.
        camera_info: Dict containing camera calibration from LiberoEnv.get_camera_info():
            - 'intrinsic': 3x3 camera intrinsic matrix K
            - 'camera_to_world': 4x4 transform from camera to world frame
            - 'image_height': image height in pixels
            - 'image_width': image width in pixels
        current_ee_pos: Current end-effector position [x, y, z] in world frame.
                        If None, returns target position scaled by gain.
        gain: Proportional gain for the delta action (0-1).

    Returns:
        delta_action: np.ndarray of shape (7,) containing [dx, dy, dz, 0, 0, 0, gripper]
    """
    # undo horizontal flip (libero base env returns double-flipped images as this is what the policy is expecting)
    segmented_frame = segmented_frame[:, ::-1] 
    depth_observation = depth_observation[0, :, ::-1, 0]

    # Handle batch dimension in depth observation
    assert depth_observation.shape == segmented_frame.shape
    assert len(depth_observation.shape) == 2

    h, w = segmented_frame.shape[:2]

    # Find all pixel-coords of non-zero points
    coords = np.argwhere(segmented_frame > 0)

    # centroid_row = int(np.mean(coords[:, 0]))
    # centroid_col = int(np.mean(coords[:, 1]))


    world_to_camera = get_camera_transform_matrix(
        env.envs[0].sim,
        "agentview",
        h,
        w,
    )
    camera_to_world = np.linalg.inv(world_to_camera)

    pixels = np.array([coords[:, 0], coords[:, 1]])
    target_pos_world = transform_from_pixels_to_world(
        pixels=pixels.T,
        depth_map=depth_observation,
        camera_to_world_transform=camera_to_world
    )

    target_pos_world = np.array(
        [
            (np.percentile(target_pos_world[:, 0], 95) + np.percentile(target_pos_world[:, 0], 5)) / 2,
            (np.percentile(target_pos_world[:, 1], 95) + np.percentile(target_pos_world[:, 1], 5)) / 2,
            (np.percentile(target_pos_world[:, 2], 95) + np.percentile(target_pos_world[:, 2], 5)) / 2,
        ]
    )


    # Compute delta action
    if current_ee_pos is not None:
        # Compute direction from EE to target
        delta = target_pos_world - current_ee_pos
        # Apply gain
        delta = delta * gain
        # Clip maximum delta to prevent large jumps
        max_delta = 0.1  # Maximum 5cm per step
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_delta:
            delta = delta / delta_norm * max_delta
    else:
        # Return scaled target position as action (fallback)
        assert False, "this delta computation is incorrect, I think."
        delta = target_pos_world * gain

    # Return action with zero rotations: [dx, dy, dz, 0, 0, 0, gripper]
    action = np.zeros(7, dtype=np.float32)
    action[:3] = delta

    if SAVE_FRAMES:
        # Debug: save segmented_frame with centroid overlay and world coordinates
        debug_img = cv2.cvtColor(segmented_frame * 255, cv2.COLOR_GRAY2BGR)
        # Add text showing world coordinates
        cv2.putText(
            debug_img,
            f"World: ({target_pos_world[0]:.3f}, {target_pos_world[1]:.3f}, {target_pos_world[2]:.3f})",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        if current_ee_pos is not None:
            cv2.putText(
                debug_img,
                f"EE: ({current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )
        frame_idx = len(os.listdir(debug_dir))
        cv2.imwrite(os.path.join(debug_dir, f"segmented_frame_{frame_idx:05d}.png"), debug_img)
        
        # Save depth image as greyscale
        depth_vis = depth_observation
        # Normalize depth to 0-255 range for visualization
        depth_min, depth_max = depth_vis.min(), depth_vis.max()
        if depth_max > depth_min:
            depth_normalized = ((depth_vis - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_vis, dtype=np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f"depth_frame_{frame_idx:05d}.png"), depth_normalized)

    return action


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    segmented_frames = None
    
    rnd_path = None # "outputs/train/2025-12-15/15-09-16_pusht_diffusion_fine_l1_RND_OOD/rnd/rnd.pth"
    if rnd_path:
        rnd = RND(config = None)
        rnd.load_state_dict(torch.load())
        rnd.eval()
        rnd.cuda()
        policy.rnd=rnd
        rnd_scores = []
        max_steps = 300

    system2 = system2_module.System2()
    # Initialize SAM3 streaming client for segmentation
    if system2.has_sam3_stage():
        sam3_client = SAM3StreamClient()

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)

    # Reset System2 for new episode
    system2.reset()

    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    action = torch.zeros((1,7))
    success_prediction = np.array([[-1.0]*env.num_envs]).T

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        if rnd_path:
            loop_start = time.time()
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))
            
        # Send RGB observation to SAM3 and receive segmented frame
        if system2.has_sam3_stage():
            # Reset SAM3 at episode start
            if step == 0:
                sam3_client.send_model_reset()
                sam3_client.drain_stale_messages()

            current_stage = system2.get_current_stage()

            # Build per-camera prompts dict
            camera_prompts = {
                key: value
                for key, value in (
                    ("3rd_person", current_stage.sam3_third_person_prompt),
                    ("1st_person", current_stage.sam3_first_person_prompt),
                )
                if value is not None
            }

            sam3_client.send_frame_batched(
                observation,
                sam3_stage=current_stage.sam3_stage,
                prompt=camera_prompts
            )
            segmented_frames = sam3_client.receive_segmented_frame()

        # Check System2 stage advancement every step (independent of SAM3)
        _ = system2.check_and_advance(
            observation=observation,
            segmented_frames=segmented_frames,
            success_prediction=success_prediction
        )

        # Determine mode from current stage
        current_stage = system2.get_current_stage()

        assert current_stage.mode in ["sam3", "policy", "home"], f"Unknown stage mode: {current_stage.mode}"

        # Compute vs action
        if current_stage.mode == "sam3" and segmented_frames is not None:
            # Get camera from current stage
            stage_camera = current_stage.vs_camera
            if stage_camera is None:
                raise ValueError("SAM3 stage must have a camera specified")

            depth_key = VISUAL_SERVOING_SETTINGS[stage_camera]["depth_name"]
            depth_obs = observation[depth_key].cpu().numpy()

            servoing_fnc = globals()[VISUAL_SERVOING_SETTINGS[stage_camera]["servoing_fnc"]]


            if stage_camera == "3rd_person":
                # Query camera_info dynamically for this stage's camera
                camera_info = env.envs[0].get_camera_info(
                    VISUAL_SERVOING_SETTINGS[stage_camera]["libero_camera_name"]
                )

                current_ee_pos = None
                if "observation.state" in observation:
                    state = observation["observation.state"].cpu().numpy()[0]
                    current_ee_pos = state[:3]  # First 3 elements are EE position

                action_np = servoing_fnc(
                    env=env,
                    segmented_frame=segmented_frames.get(stage_camera),
                    depth_observation=depth_obs,
                    camera_info=camera_info,
                    current_ee_pos=current_ee_pos,
                    gain=np.array([6, 6, 2])
                )
            elif stage_camera == "1st_person":
                action_np = servoing_fnc(
                    segmented_frame=segmented_frames.get(stage_camera),
                    depth_observation=depth_obs,
                    current_ee_pos=None,  # Could extract from observation if available
                    camera_intrinsics=None,  # Uses defaults
                    gain=2,
                )
            else:
                raise ValueError(f"Unknown visual servoing camera: {stage_camera}")
            action = torch.from_numpy(action_np).unsqueeze(0)  # Add batch dim

        # Use policy if in policy mode
        elif current_stage.mode == "policy":
            # Infer "task" from attributes of environments.
            # TODO: works with SyncVectorEnv but not AsyncVectorEnv
            observation = add_envs_task(env, observation)
            # use custom tasks if trained on them:
            # policy.diffusion.recover_task_dict()
            observation["task"] = current_stage.policy_instruction
            observation["task_index"] = current_stage.policy_task_index
            # here needs to go additinoal data augmentation if any
            observation = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)
        elif current_stage.mode == "home":
            action_np = move_to_home(
                observation=observation,
                gain=0.5,
            )
            action = torch.from_numpy(action_np).unsqueeze(0)  # Add batch dim
        
        if rnd_path:
            rnd_score = policy.predict_rnd(observation)
            rnd_scores.append(float(rnd_score))

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action
        # assume success prediciton as last action dim for libero env
        if action_numpy.shape[1] == 8:
            success_prediction = action_numpy[:, -1]
            action_numpy = action_numpy[:, :7]
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        if rnd_path:
            # Render
            render_img = cv2.cvtColor(observation["pixels"][0], cv2.COLOR_RGB2BGR)

            # Create simple graph matching render_img width
            width = render_img.shape[1]
            graph = np.ones((150, width, 3), dtype=np.uint8) * 255

            # Draw RND scores as line
            scores = np.array(rnd_scores)
            if len(scores) > 1:
                min_score = scores.min()
                max_score = scores.max()
                
                for i in range(len(scores) - 1):
                    x1 = int(i * width / max_steps)
                    x2 = int((i + 1) * width / max_steps)
                    y1 = int(130 - (scores[i] - min_score) / (max_score - min_score + 1e-10) * 120)
                    y2 = int(130 - (scores[i + 1] - min_score) / (max_score - min_score + 1e-10) * 120)
                    cv2.line(graph, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Show current, min, max in scientific notation
                cv2.putText(graph, f"RND: {rnd_scores[-1]:.2e}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                cv2.putText(graph, f"Max: {max_score:.2e}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
                cv2.putText(graph, f"Min: {min_score:.2e}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

            # Stack and show
            combined = np.vstack([render_img, graph])
            cv2.imshow("pusht", combined)
            cv2.waitKey(1)
            
            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < 1.0 / env.envs[0].unwrapped.metadata["render_fps"]:
                time.sleep(1.0 / env.envs[0].unwrapped.metadata["render_fps"] - elapsed)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    # Cleanup SAM3 client
    if 'sam3_client' in locals():
        sam3_client.close()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    only_render_failures: bool = False
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for i, (stacked_frames, done_index) in enumerate(zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            )):
                if n_episodes_rendered >= max_episodes_rendered:
                    break
                if only_render_failures:
                    # skip saving if episode was successful -> only save videos of failed episodes
                    if rollout_data["success"][i][done_index]:
                        # just a check that I understood the code correctly
                        assert batch_successes[i], "This should not be triggered. I most likely misunderstood the code"
                        continue
                    assert not batch_successes[i], "This should not be triggered. I most likely misunderstood the code"

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        # Below renders only last frame
                        # stacked_frames[done_index: done_index + 1],  # only save last frame - its enough to investigate failure case
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.eval.max_episodes_rendered,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            only_render_failures=cfg.eval.only_render_failures
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    only_render_failures: bool = False
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        only_render_failures=only_render_failures
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    only_render_failures: bool = False
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        only_render_failures=only_render_failures
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    only_render_failures: bool = False
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        only_render_failures=only_render_failures
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    eval_main()


if __name__ == "__main__":
    main()
