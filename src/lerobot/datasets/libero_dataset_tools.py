#!/usr/bin/env python
"""
Fast dataset task analysis using LeRobot episode metadata.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def analyze_dataset_tasks(dataset, output_dir="dataset_task_analysis"):
    """
    Fast analysis using LeRobot episode metadata.

    Args:
        dataset: LeRobotDataset with meta.episodes
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Using fast analysis with episode metadata...\n")

    episodes = dataset.meta.episodes

    print(f"Number of episodes: {len(episodes)}\n")

    task_to_indices = defaultdict(list)
    task_to_episodes = defaultdict(set)
    task_to_episode_list = defaultdict(list)

    # Iterate through episodes (1693 instead of 273465!)
    for ep_idx in tqdm(range(len(episodes)), desc="Processing episodes"):
        # tasks is a list, get the first element
        task = episodes[ep_idx]["tasks"][0]
        start_idx = episodes[ep_idx]["dataset_from_index"]
        end_idx = episodes[ep_idx]["dataset_to_index"]

        # Add all sample indices for this episode
        sample_indices = list(range(start_idx, end_idx))
        task_to_indices[task].extend(sample_indices)
        task_to_episodes[task].add(ep_idx)
        task_to_episode_list[task].append(ep_idx)

    # Convert to regular dicts
    task_to_indices = dict(task_to_indices)
    task_to_episode_counts = {
        task: len(episodes) for task, episodes in task_to_episodes.items()
    }
    task_to_episode_list = dict(task_to_episode_list)

    # Print results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nTotal unique tasks found: {len(task_to_indices)}")
    print(
        f"Total samples: {sum(len(indices) for indices in task_to_indices.values())}\n"
    )

    sorted_tasks = sorted(
        task_to_indices.items(), key=lambda x: len(x[1]), reverse=True
    )

    for task, indices in sorted_tasks:
        num_samples = len(indices)
        num_episodes = task_to_episode_counts[task]
        print(f"Task: {task}")
        print(f"  - Samples: {num_samples}")
        print(f"  - Episodes: {num_episodes}")
        print(f"  - Samples per episode (avg): {num_samples / num_episodes:.2f}")
        print()

    results = {
        "task_to_indices": task_to_indices,
        "task_to_episode_counts": task_to_episode_counts,
        "task_to_episode_list": task_to_episode_list,"total_samples": len(dataset),
        "total_unique_tasks": len(task_to_indices),
    }

    # Save files
    pickle_path = output_dir / "task_analysis.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"✓ Saved pickle file to: {pickle_path}")

    json_path = output_dir / "task_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON file to: {json_path}")

    summary_path = output_dir / "task_summary.txt"
    with open(summary_path, "w") as f:
        f.write("DATASET TASK ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Total unique tasks: {len(task_to_indices)}\n\n")

        for task, indices in sorted_tasks:
            num_samples = len(indices)
            num_episodes = task_to_episode_counts[task]
            f.write(f"Task: {task}\n")
            f.write(f"  Samples: {num_samples}\n")
            f.write(f"  Episodes: {num_episodes}\n")
            f.write(f"  Avg samples/episode: {num_samples / num_episodes:.2f}\n\n")

    print(f"✓ Saved summary to: {summary_path}")
    print("\n" + "=" * 80)

    return results


def load_task_analysis(output_dir="dataset_task_analysis"):
    """
    Load previously saved task analysis.

    Args:
        output_dir: Directory where analysis was saved

    Returns:
        dict: Contains task_to_indices and task_to_episode_counts
    """
    output_dir = Path(output_dir)
    pickle_path = output_dir / "task_analysis.pkl"

    if not pickle_path.exists():
        raise FileNotFoundError(f"Analysis file not found at {pickle_path}")

    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    print(f"Loaded task analysis from {pickle_path}")
    print(f"  - Total tasks: {results['total_unique_tasks']}")
    print(f"  - Total samples: {results['total_samples']}")

    return results
