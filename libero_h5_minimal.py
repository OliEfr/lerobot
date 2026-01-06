import numpy as np
from pathlib import Path
from h5py import File
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Configuration ---
# Update these paths to your local environment
dataset_save_name = "libero_10_id4_pick_50_2"

INPUT_H5 = Path("/home/admin_07/project_repos/LIBERO/libero/datasets/libero_10_regenerated_openvla/demo.hdf5")
OUTPUT_PATH = Path(f"./libero_datasets/{dataset_save_name}")
REPO_ID = f"OliverHausdoerfer/{dataset_save_name}"
FPS = 20

LIBERO_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "ax", "ay", "az", "g1", "g2"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "ax", "ay", "az", "gripper"]},
    },
}

def convert_dataset():
    # 1. Initialize the v3 Dataset
    # We use 'create' which initializes the LeRobotDatasetMetadata and empty structures
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=LIBERO_FEATURES,
        root=OUTPUT_PATH,
        robot_type="franka",
    )

    with File(INPUT_H5, "r") as f:
        # LIBERO datasets often store the task string in attributes
        task_label = "test_task"
        if isinstance(task_label, bytes):
            task_label = task_label.decode("utf-8")

        demos = list(f["data"].values())
        
        for demo in tqdm(demos, desc="Converting Episodes"):
            demo_len = len(demo["obs/agentview_rgb"])

            actions = np.array(demo["actions"])

            # State preprocessing
            ee_states = np.array(demo["obs/ee_states"], dtype=np.float32)
            gripper_states = np.array(demo["obs/gripper_states"], dtype=np.float32)
            combined_state = np.concatenate([ee_states, gripper_states], axis=1)

            # 2. Add frames to the internal buffer
            for i in range(demo_len):
                frame_data = {
                    "observation.images.image": demo["obs/agentview_rgb"][i],
                    "observation.images.image2": demo["obs/eye_in_hand_rgb"][i],
                    "observation.state": combined_state[i],
                    "action": actions[i].astype(np.float32),
                    "task": task_label, # Required for task_index mapping in v3
                }
                dataset.add_frame(frame_data)

            # 3. Save the episode
            # This handles task indexing, stats computation, and parquet writing
            dataset.save_episode()

    # 4. CRITICAL: Finalize the dataset
    # This closes parquet writers and writes footer metadata
    dataset.finalize()
    print(f"Dataset successfully converted to {OUTPUT_PATH}")

    # Optional: push to hub
    dataset.push_to_hub()

if __name__ == "__main__":
    convert_dataset()