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
# keys
import os
from pathlib import Path
import torch

from huggingface_hub.constants import HF_HOME

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OBS_DEPTH = OBS_STR + ".depth"
OBS_DEPTHS = OBS_DEPTH + "s"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_TASK_IDS = OBS_STR + ".task_ids"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"

ACTION = "action"
REWARD = "next.reward"
TRUNCATED = "next.truncated"
DONE = "next.done"

ROBOTS = "robots"
TELEOPERATORS = "teleoperators"

# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
    )

# cache dir
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

# calibration dir
default_calibration_path = HF_LEROBOT_HOME / "calibration"
HF_LEROBOT_CALIBRATION = Path(os.getenv("HF_LEROBOT_CALIBRATION", default_calibration_path)).expanduser()


# streaming datasets
LOOKBACK_BACKTRACKTABLE = 100
LOOKAHEAD_BACKTRACKTABLE = 100

# openpi
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38  # TODO(pepijn): Modify this when extending support to fp8 models


COVER_GREEN_T_START_X_VIS = 25
COVER_GREEN_T_START_Y_VIS = 30
COVER_GREEN_T_END_X_VIS = 65
COVER_GREEN_T_END_Y_VIS = 70

VIS_TO_STATE_SCALE = 512 / 96

COVER_GREEN_T_START_X_STATE = int(COVER_GREEN_T_START_X_VIS * VIS_TO_STATE_SCALE)
COVER_GREEN_T_START_Y_STATE = int(COVER_GREEN_T_START_Y_VIS * VIS_TO_STATE_SCALE)
COVER_GREEN_T_END_X_STATE = int(COVER_GREEN_T_END_X_VIS * VIS_TO_STATE_SCALE)
COVER_GREEN_T_END_Y_STATE = int(COVER_GREEN_T_END_Y_VIS * VIS_TO_STATE_SCALE)

NUM_TASKS = 5 # hardcoded for now; determines size of one-hot vector
TASK_ONEHOT_LOOKUP = torch.eye(NUM_TASKS, dtype=torch.float32, device="cuda")

# Visual servoing camera settings
VISUAL_SERVOING_SETTINGS = {
    "1st_person": {
        "lerobot_camera_name": "observation.images.image2",
        "libero_camera_name": "robot0_eye_in_hand",
        "depth_name": "observation.depths.depth2",
        "servoing_fnc": "VS_first_person"
    },
    "3rd_person": {
        "lerobot_camera_name": "observation.images.image",
        "libero_camera_name": "agentview",
        "depth_name": "observation.depths.depth",
        "servoing_fnc": "VS_third_person",
    }
}