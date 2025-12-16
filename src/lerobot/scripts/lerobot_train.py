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
import sys
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from torch.utils.data import SubsetRandomSampler

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

from lerobot.datasets.libero_dataset_tools import analyze_dataset_tasks

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        print("Debugging in VSCode, disabling wandb.")
        cfg.wandb.enable = False

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    if hasattr(cfg.policy, "partial_green_t_cover_processor") and cfg.policy.partial_green_t_cover_processor:
        assert cfg.env.task == "PushT-v0", "This processor is only compatible with PushT-v0"
        assert "environment_state" in cfg.env.features and "observation.environment_state" in cfg.policy.input_features, "You probably want to use the environment state in your env and policy when using partial_green_t_cover_processor."
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if (
        "libero" in cfg.dataset.repo_id
        and hasattr(cfg, "task_to_solve")
        and cfg.task_to_solve is not None
    ):
        libero_stats = analyze_dataset_tasks(dataset, output_dir="libero_dataset_stats")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        assert not "libero_stats" in locals(), "still need to handle this case"
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
        if "libero_stats" in locals():
            sample_idx_to_use = libero_stats["task_to_indices"][cfg.task_to_solve]
            sampler = SubsetRandomSampler(sample_idx_to_use)
            shuffle = False # SubsetRandomSampler shuffles by default already

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames, # NOTE this is not correct incase sampler is used
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    # train rnd
    if hasattr(cfg.policy, "use_rnd") and cfg.policy.use_rnd:
        train_len = int(len(dataset) * 0.95)
        test_len = len(dataset) - train_len
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])

        dataloader_train = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=True and not cfg.dataset.streaming,
            sampler=None,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

        dataloader_test = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=None,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

        # Early stopping setup
        best_test_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(0, 50):
            epoch_loss_train = 0
            epoch_loss_test = 0

            # train
            for batch in dataloader_train:
                batch = preprocessor(batch)
                loss = policy.diffusion.rnd.train_on_batch(batch)
                epoch_loss_train += loss.item()
            epoch_loss_train /= len(dataloader_train)

            # test
            for batch in dataloader_test:
                batch = preprocessor(batch)
                with torch.no_grad():
                    loss = policy.diffusion.rnd.compute_loss(batch).mean()
                epoch_loss_test += loss.item()
            epoch_loss_test /= len(dataloader_test)

            print(f"[RND] Epoch: {epoch}, Train Loss: {epoch_loss_train}, Test Loss: {epoch_loss_test}")
            if wandb_logger:
                wandb_logger.log_dict({"rnd_train_loss": epoch_loss_train, "rnd_test_loss": epoch_loss_test}, epoch, mode="train")

            # Early stopping
            if epoch_loss_test < best_test_loss:
                best_test_loss = epoch_loss_test
                patience_counter = 0
                print(f"[RND] New best test loss: {best_test_loss:.6f}")
            elif epoch > 5:
                patience_counter += 1
                print(f"[RND] No improvement for {patience_counter} epoch(s)")

            if patience_counter >= patience:
                print(f"[RND] Early stopping triggered at epoch {epoch}. Best test loss: {best_test_loss:.6f}")
                break
            
    # save rnd model separately
    rnd_save_dir = cfg.output_dir / "rnd"
    rnd_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.diffusion.rnd.state_dict(), rnd_save_dir / "rnd.pth")


    if cfg.policy.train_only_rnd:
        assert cfg.policy.use_rnd, "train_only_rnd requires use_rnd to be True"
        return


    # all_actions = []
    # all_states = []
    # all_env_states = []

    # Iterate through the DataLoader
    # for batch in tqdm(dataloader, desc=f"Processing .."):
        # all_actions.append(batch["action"][:,0,:] - batch["observation.state"][:,0,:])  # appends (batch_size, action_dim)
        # all_states.append(batch["observation.state"][:, 0, :])  # (batch_size, state_dim)
        # all_env_states.append(batch["observation.environment_state"][:, 0, :])  # (batch_size, env_state_dim)

    # Concatenate all batches
    # all_actions_cat = torch.cat(all_actions, dim=0).clone()  # (dataset_size, action_dim)
    # all_states_cat = torch.cat(all_states, dim=0)  # (dataset_size, state_dim)
    # all_env_states_cat = torch.cat(all_env_states, dim=0)  # (dataset_size, env_state_dim)

    # # Calculate magnitudes
    # action_magnitudes = np.linalg.norm(all_actions_cat.cpu.numpy(), axis=-1)
    # action_magnitudes = torch.linalg.norm(all_actions_cat, axis=-1)
    # non_zero_mask = action_magnitudes > 0
    # magnitudes_nonzero = action_magnitudes[non_zero_mask]

    # weights = torch.exp(-2*(action_magnitudes-1))+0.1
    # weights = -3 * torch.atan(action_magnitudes - 5) + 5.7
    # weights_mean= weights.mean()
    # policy.diffusion.loss_weight_mean = weights_mean

    # # Concatenate state and environment state
    # combined_state = all_states_cat
    # # combined_state = torch.cat([all_states_cat, all_env_states_cat], dim=-1)  # (dataset_size, state_dim + env_state_dim)
    # combined_state_nonzero = combined_state[non_zero_mask].cpu().numpy()

    # # Apply PCA to reduce to 2D
    # print("Applying PCA to combined state...")
    # pca = PCA(n_components=2)
    # state_compressed = pca.fit_transform(combined_state_nonzero)
    # print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    # print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # print("Applying UMAP to combined state...")
    # umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    # state_compressed = umap.fit_transform(combined_state_nonzero)

    # # tsne = TSNE(
    # #     n_components=2,
    # #     random_state=42,
    # #     perplexity=50,       # Higher for larger datasets (30-100 is good)
    # #     learning_rate='auto',
    # #     init='pca',          # PCA initialization is faster
    # #     n_jobs=-1
    # # )
    # # state_tsne = tsne.fit_transform(combined_state_nonzero)
    # # print("t-SNE complete!")

    # # Non-zero action statistics
    # print("\nNon-zero actions only:")
    # print(f"  Count: {len(magnitudes_nonzero)} ({100*non_zero_mask.sum()/len(non_zero_mask):.2f}% of total)")
    # print(f"  Min magnitude:  {magnitudes_nonzero.min():.6f}")
    # print(f"  Max magnitude:  {magnitudes_nonzero.max():.6f}")
    # print(f"  Mean magnitude: {magnitudes_nonzero.mean():.6f}")
    # print(f"  Std magnitude:  {magnitudes_nonzero.std():.6f}")
    # print(f"  Median magnitude: {np.median(magnitudes_nonzero):.6f}")

    # # Create visualization
    # fig, ax = plt.subplots(figsize=(10, 8))

    # # Create scatter plot with color representing action magnitude
    # scatter = ax.scatter(
    #     state_compressed[:, 0],
    #     state_compressed[:, 1],
    #     c=torch.asinh(torch.tensor(magnitudes_nonzero)/2).cpu().numpy() , # scale using asinh
    #     cmap='viridis',
    #     s=10,
    #     alpha=0.6,
    #     edgecolors='none'
    # )

    # # Add colorbar
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('Action Magnitude', fontsize=12)

    # # Labels and title
    # ax.set_xlabel(f'1', fontsize=12)
    # ax.set_ylabel(f'2', fontsize=12)
    # ax.set_title('Action Magnitude vs State compressed', fontsize=14, fontweight='bold')
    # ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

    # print(f"Total non-zero actions plotted: {len(magnitudes_nonzero)}")
    # print(f"Action magnitude range: [{magnitudes_nonzero.min():.3f}, {magnitudes_nonzero.max():.3f}]")

    # # --- Parameters for Scaling and Training ---
    # K = 1.0       # Hyperparameter for overall scale of beta
    # EPSILON = 0.05  # Numerical stability constant for inverse magnitude
    # NN_HIDDEN_SIZE = 512
    # NN_EPOCHS = 50
    # NN_BATCH_SIZE = 128
    # NN_LEARNING_RATE = 1e-3

    # # --- 1. Data Preparation for Regressor (Directly using 2D state) ---

    # # Input data: The 2D states (X_train)
    # # combined_state_nonzero contains the 2D state for non-zero actions
    # X_train_np = combined_state_nonzero

    # # --- 1. NORMALIZE INPUTS ---
    # input_scaler = StandardScaler()
    # X_train_scaled = input_scaler.fit_transform(X_train_np)
    # nn_X_train = torch.tensor(X_train_scaled, dtype=torch.float32)

    # # --- 2. LOG-TRANSFORM TARGETS ---
    # # This makes the range manageable: log(0.0027) ≈ -5.9, log(19.99) ≈ 3.0
    # target_beta = K / (magnitudes_nonzero + EPSILON)
    # target_beta_log = np.log(target_beta)

    # nn_y_train = torch.tensor(target_beta_log, dtype=torch.float32).unsqueeze(1)

    # # Create DataLoader for training
    # nn_ds = TensorDataset(nn_X_train, nn_y_train)
    # nn_loader = DataLoader(nn_ds, batch_size=NN_BATCH_SIZE, shuffle=True)
    # print(f"Regressor dataset size: {len(nn_X_train)}")
    # print(f"Target beta range (log-space): [{nn_y_train.min().item():.3f}, {nn_y_train.max().item():.3f}]")
    # print(f"Target beta range (original): [{target_beta.min():.3f}, {target_beta.max():.3f}]")

    # # --- 2. Define and Train the Neural Network Regressor ---

    # class ScaleRegressor(nn.Module):
    #     def __init__(self, input_dim, hidden_size, dropout_rate=0.2):
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.Linear(input_dim, hidden_size),
    #             nn.ReLU(),  # ReLU is more stable than LeakyReLU for this
    #             nn.Dropout(dropout_rate),  # Add dropout
    #             nn.Linear(hidden_size, hidden_size),
    #             nn.ReLU(),  # ReLU is more stable than LeakyReLU for this
    #             nn.Dropout(dropout_rate),  # Add dropout
    #             nn.Linear(hidden_size, hidden_size),
    #             nn.ReLU(),
    #             nn.Dropout(dropout_rate),  # Add dropout
    #             nn.Linear(hidden_size, 1),
    #         )

    #     def forward(self, x):
    #         return self.net(x)

    # # Initialize model, loss, and optimizer
    # nn_model = ScaleRegressor(input_dim=2, hidden_size=NN_HIDDEN_SIZE)
    # nn_criterion = nn.MSELoss()
    # nn_optimizer = optim.Adam(nn_model.parameters(), lr=NN_LEARNING_RATE)

    # # Training loop with epoch logging
    # print("\nTraining Scale Regressor...")
    # nn_model.train()
    # for epoch in tqdm(range(NN_EPOCHS), desc="Training Regressor"):
    #     epoch_loss = 0.0
    #     batch_count = 0

    #     for states, betas in nn_loader:
    #         nn_optimizer.zero_grad()
    #         predicted_betas = nn_model(states)
    #         loss = nn_criterion(predicted_betas, betas)
    #         loss.backward()
    #         nn_optimizer.step()

    #         epoch_loss += loss.item()
    #         batch_count += 1

    #     avg_epoch_loss = epoch_loss / batch_count

    #     # Log every 10 epochs or first 5 epochs
    #     if epoch % 10 == 0 or epoch < 5:
    #         print(f"Epoch {epoch:3d}/{NN_EPOCHS}, Loss: {avg_epoch_loss:.6f}")

    # print(f"Final training loss: {avg_epoch_loss:.6f}")

    # # --- 3. Generate and Plot Learned Scaling Function ---

    # # Set model to evaluation mode
    # nn_model.eval()

    # X_min, X_max = 0, 500
    # Y_min, Y_max = 0, 500

    # grid_points = 50 # 50x50 grid = 2500 points
    # xx, yy = np.meshgrid(
    #     np.linspace(X_min, X_max, grid_points),
    #     np.linspace(Y_min, Y_max, grid_points)
    # )

    # # Prepare grid points for NN inference
    # grid_states = np.vstack([xx.ravel(), yy.ravel()]).T

    # # IMPORTANT: Scale the grid states using the same scaler
    # grid_states_scaled = input_scaler.transform(grid_states)
    # grid_states_tensor = torch.tensor(grid_states_scaled, dtype=torch.float32)

    # # Infer the learned scale (beta) for each grid point
    # with torch.no_grad():
    #     learned_beta_log = nn_model(grid_states_tensor).squeeze().numpy()
    #     # IMPORTANT: Transform back from log-space to original scale
    #     learned_beta = np.exp(learned_beta_log)

    # # Reshape the results to the grid dimensions
    # learned_beta_grid = learned_beta.reshape(xx.shape)

    # print(f"\nLearned beta statistics:")
    # print(f"  Min: {learned_beta.min():.6f}")
    # print(f"  Max: {learned_beta.max():.6f}")
    # print(f"  Mean: {learned_beta.mean():.6f}")
    # print(f"  Should be close to target range: [{target_beta.min():.3f}, {target_beta.max():.3f}]")

    # # --- 4. Plotting (Original Data + Learned Scale Overlay) ---

    # fig, ax = plt.subplots(figsize=(10, 8))

    # # 1. Plot the Original Data (Action Magnitudes)
    # # Use the un-PCA'd state data (X_train_np)
    # scatter = ax.scatter(
    #     state_compressed[:, 0],
    #     state_compressed[:, 1],
    #     # Use asinh scaling for visual clarity of magnitude
    #     c=torch.asinh(torch.tensor(magnitudes_nonzero)/2).cpu().numpy(),
    #     cmap='viridis',
    #     s=10,
    #     alpha=0.6,
    #     edgecolors='none',
    #     zorder=1 # Ensure scatter points are visually distinct
    # )

    # # 2. Plot the Learned Scale (Beta) as an Overlay (Contour Plot)
    # contour = ax.contourf(
    #     pca.transform(xx), pca.transform(yy), learned_beta_grid,
    #     levels=15, # Number of contour levels
    #     cmap='plasma', # A different colormap to clearly distinguish the overlay
    #     alpha=0.4, # Transparency to see the scatter plot underneath
    #     zorder=0 # Ensure contour is below the scatter points
    # )

    # # Create a separate colorbar for the learned scale (Beta)
    # cbar_beta = plt.colorbar(contour, ax=ax, pad=0.1)
    # cbar_beta.set_label(r'Learned State Scale $\beta(\mathbf{S})$', fontsize=12)

    # # Colorbar for Action Magnitude (Original data)
    # cbar_mag = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
    # cbar_mag.set_label('Action Magnitude (asinh scaled)', fontsize=12)

    # # Labels and title
    # ax.set_xlabel(f'State Dimension 1', fontsize=12)
    # ax.set_ylabel(f'State Dimension 2', fontsize=12)
    # ax.set_title('Action Magnitude (Scatter) vs Learned Scale $\\beta$ (Contour)', fontsize=14, fontweight='bold')
    # ax.grid(True, alpha=0.3)
    # ax.set_xlim(X_min, X_max)
    # ax.set_ylim(Y_min, Y_max)

    # plt.tight_layout()
    # plt.show()

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        # rel_actions = (
        #     (batch["action"] - batch["observation.state"])
        #     .clone()
        #     .detach()
        # )
        # rel_actions_magnitudes = torch.linalg.norm(rel_actions, axis=-1)

        # policy.diffusion.loss_scales = (
        #     -3 * torch.atan(rel_actions_magnitudes - 5) + 5.7
        # ).clone().detach()
        # torch.exp(-2 * (rel_actions_magnitudes - 1)) + 0.1

        # modify obs to only include indices n_obs_steps expected by policy
        # batch["observation.state"] = batch["observation.state"][:,:2,:]
        # batch["observation.environment_state"] = batch["observation.environment_state"][:,:2,:]

        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                for p in preprocessor:
                    if hasattr(p, "get_metrics"):
                        wandb_log_dict.update(p.get_metrics())
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=cfg.eval.max_episodes_rendered,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames, # NOTE this is not correct incase sampler is used
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    for p in preprocessor:
                        if hasattr(p, "get_metrics"):
                            wandb_log_dict.update(p.get_metrics())
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    if cfg.eval.max_episodes_rendered > 0:
                        wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    main()
