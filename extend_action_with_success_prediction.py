import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_stats
from lerobot.utils.utils import init_logging


@dataclass
class ExtendActionConfig:
    """Configuration for extending action dimensions with success prediction.

    Args:
        repo_id: Repository ID of the source dataset (e.g., "libero_10_id4_pick_150_merged")
        root: Root directory containing the dataset (e.g., "libero_datasets")
        push_to_hub: Whether to push the new dataset to HuggingFace Hub
        new_suffix: Suffix to append to repo_id for the new dataset
        hub_user: HuggingFace Hub username/organization to push to
        success_window: Number of frames at the end of each episode to mark as success
    """
    repo_id: str = "libero_10_id4_yellow_white_mug_pick_place_merged"
    root: str = "libero_datasets"
    push_to_hub: bool = True
    new_suffix: str = "_SP"
    hub_user: str = "OliverHausdoerfer"
    success_window: int = 3


def create_extended_actions(dataset: LeRobotDataset, episode_lengths: dict[int, int], success_window: int) -> np.ndarray:
    """Create extended action array with success prediction dimension.

    Pre-computes all extended actions by loading the original actions and
    appending the success prediction dimension.

    Args:
        dataset: LeRobot dataset to extend
        episode_lengths: Dictionary mapping episode_index to episode length
        success_window: Number of frames at the end of each episode to mark as success

    Returns:
        Extended action array with shape (total_frames, original_action_dim + 1)
        Last dimension is 1.0 for last success_window frames per episode, -1.0 otherwise
    """
    logging.info("Loading original actions from dataset...")

    total_frames = dataset.meta.total_frames
    original_action_dim = dataset.meta.features["action"]["shape"][0]

    # Initialize extended actions array
    extended_actions = np.zeros((total_frames, original_action_dim + 1), dtype=np.float32)

    # Load original actions and compute success prediction for each frame
    logging.info(f"Processing {total_frames} frames...")

    for idx in range(total_frames):
        frame = dataset.hf_dataset[idx]

        # Get original action
        original_action = np.array(frame["action"])

        # Get episode info
        episode_idx = frame["episode_index"].item() if hasattr(frame["episode_index"], "item") else frame["episode_index"]
        frame_idx = frame["frame_index"].item() if hasattr(frame["frame_index"], "item") else frame["frame_index"]

        # Determine if in last success_window frames
        episode_length = episode_lengths[episode_idx]
        is_success_frame = 1.0 if frame_idx >= (episode_length - success_window) else -1.0 # use -1 and 1 for better normalization

        # Store extended action
        extended_actions[idx, :original_action_dim] = original_action
        extended_actions[idx, original_action_dim] = is_success_frame

    logging.info(f"Extended actions created with shape {extended_actions.shape}")
    return extended_actions


@parser.wrap()
def extend_action_with_success_prediction(cfg: ExtendActionConfig) -> None:
    """Main function to extend action dimensions with success prediction.

    Args:
        cfg: Configuration object containing repo_id, root, and other settings
    """
    # Step 1: Load the dataset
    dataset_root = Path(cfg.root) / cfg.repo_id
    logging.info(f"Loading dataset from {dataset_root}")
    try:
        dataset = LeRobotDataset(
            repo_id=cfg.repo_id,
            root=dataset_root,
        )
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    logging.info(f"Dataset loaded: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")

    # Validate that action feature exists
    if "action" not in dataset.meta.features:
        raise ValueError(f"Dataset does not contain 'action' feature. Available features: {list(dataset.meta.features.keys())}")

    # Get current action shape
    original_action_shape = dataset.meta.features["action"]["shape"]
    logging.info(f"Current action shape: {original_action_shape}, extending to ({original_action_shape[0] + 1},)")

    # Step 2: Build episode lengths mapping
    logging.info(f"Building episode metadata for {dataset.meta.total_episodes} episodes...")
    episode_lengths = {
        idx: dataset.meta.episodes[idx]["length"]
        for idx in range(dataset.meta.total_episodes)
    }

    # Log statistics about episode lengths
    lengths = list(episode_lengths.values())
    logging.info(f"Episode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # Warn about edge cases
    short_episodes = [idx for idx, length in episode_lengths.items() if length < cfg.success_window]
    if short_episodes:
        logging.warning(f"Found {len(short_episodes)} episodes with < {cfg.success_window} frames. All frames in these episodes will have success_prediction = 1")

    # Step 3: Create extended actions
    logging.info("Creating extended action feature...")
    extended_actions = create_extended_actions(dataset, episode_lengths, cfg.success_window)

    # Build new feature info with extended shape
    new_action_shape = (original_action_shape[0] + 1,)

    # Get original motor names if they exist
    original_names = dataset.meta.features["action"].get("names", None)
    if original_names is not None and isinstance(original_names, dict) and "motors" in original_names:
        # Extend motor names with success_prediction
        original_motor_names = original_names["motors"]
        new_motor_names = original_motor_names + ["success_prediction"]
        new_names = {"motors": new_motor_names}
    else:
        new_names = None

    new_action_feature_info = {
        "dtype": "float32",
        "shape": new_action_shape,
        "names": new_names
    }

    # Step 4: Apply feature modification
    # Extract dataset name (remove any existing namespace)
    dataset_name = cfg.repo_id.split("/")[-1] if "/" in cfg.repo_id else cfg.repo_id
    new_dataset_name = f"{dataset_name}{cfg.new_suffix}"
    assert len(new_dataset_name)<55, "Dataset name too long - wandb will not allow that name during logging."

    # Create repo_id with hub namespace for pushing
    new_repo_id = f"{cfg.hub_user}/{new_dataset_name}"

    # Local output directory (without namespace)
    output_dir = Path(cfg.root) / new_dataset_name

    logging.info(f"Applying feature modification...")
    logging.info(f"Output repository: {new_repo_id}")
    logging.info(f"Output directory: {output_dir}")

    try:
        extended_dataset = modify_features(
            dataset,
            remove_features="action",
            add_features={
                "action": (extended_actions, new_action_feature_info)
            },
            output_dir=output_dir,
            repo_id=new_repo_id
        )
    except Exception as e:
        logging.error(f"Failed to modify features: {e}")
        raise

    # Log success
    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"New dataset: {extended_dataset.meta.total_episodes} episodes, {extended_dataset.meta.total_frames} frames")
    logging.info(f"Action shape updated: {original_action_shape} -> {extended_dataset.meta.features['action']['shape']}")

    # Step 5: Recompute statistics for the action feature
    logging.info("Recomputing statistics for action feature...")

    # Load existing stats
    updated_stats = extended_dataset.meta.stats.copy() if extended_dataset.meta.stats else {}

    # Compute new stats for the extended action feature
    action_stats = get_feature_stats(
        extended_actions,
        axis=0,
        keepdims=extended_actions.ndim == 1
    )

    # Update the action stats
    updated_stats["action"] = action_stats

    # Write updated stats back to disk
    write_stats(updated_stats, extended_dataset.meta.root)
    logging.info("Statistics updated successfully")

    # Step 6: Push to hub if requested
    if cfg.push_to_hub:
        logging.info(f"Pushing to HuggingFace Hub as {new_repo_id}...")
        try:
            extended_dataset.push_to_hub()
            logging.info(f"Successfully pushed to https://huggingface.co/datasets/{new_repo_id}")
        except Exception as e:
            logging.error(f"Failed to push to hub: {e}")
            logging.error("Dataset has been created locally but not pushed to hub")
            raise
    else:
        logging.info("Skipping hub push (use --push_to_hub true to push)")

    logging.info("Done!")


def main() -> None:
    """Entry point for the script."""
    init_logging()
    extend_action_with_success_prediction()


if __name__ == "__main__":
    main()
