from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


dataset = LeRobotDataset(
    repo_id="libero_10_id4_yellow_white_mug_pick_place_merged_SP",
    root="libero_datasets/libero_10_id4_yellow_white_mug_pick_place_merged_SP",
)

tasks = dataset.meta.tasks
pass