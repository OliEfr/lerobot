"""
LeRobot PushT Environment Teleoperation Recording - Mouse Control
Record episodes using mouse position with direct screen rendering
"""

import numpy as np
import torch
from pathlib import Path
import gymnasium as gym
from PIL import Image
import cv2
import time
import json

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.utils import (
    preprocess_observation,
)
from lerobot.policies.diffusion.modeling_diffusion import RND, DiffusionPolicy
from lerobot.envs.utils import (
    preprocess_observation,
)




class MouseInput:
    """Mouse input handler for controlling the agent"""
    
    def __init__(self, window_name="PushT Environment", env_size=512):
        self.window_name = window_name
        self.env_size = env_size  # Environment coordinate space (512x512)
        self.window_width = None
        self.window_height = None
        self.mouse_x = 256  # Center of 512x512 environment
        self.mouse_y = 256
        self.finish_episode = False
        self.quit_recording = False
        self.recording_started = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback function"""
        if event == cv2.EVENT_MOUSEMOVE:
            # Store raw window coordinates
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click to start recording
            self.recording_started = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to finish episode
            self.finish_episode = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle click to quit
            self.quit_recording = True
    
    def setup(self):
        """Set up the mouse callback"""
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
    
    def set_window_size(self, width: int, height: int):
        """Set the window size for coordinate scaling"""
        self.window_width = width
        self.window_height = height
    
    def get_action(self):
        """Get current mouse position as action, scaled to environment coordinates"""
        if self.window_width is None or self.window_height is None:
            # Fallback if window size not set
            return np.array([self.mouse_x, self.mouse_y], dtype=np.float32)
        
        # Scale from window coordinates to environment coordinates (512x512)
        env_x = (self.mouse_x / self.window_width) * self.env_size
        env_y = (self.mouse_y / self.window_height) * self.env_size
        
        # Clip to valid range
        env_x = np.clip(env_x, 0, self.env_size)
        env_y = np.clip(env_y, 0, self.env_size)
        
        return np.array([env_x, env_y], dtype=np.float32)
    
    def reset_flags(self):
        """Reset episode control flags"""
        self.finish_episode = False
        self.recording_started = False


class PushTTeleopRecorder:
    """Records teleoperation episodes for LeRobot PushT environment with mouse control"""

    def __init__(self, action_type, repo_id="pusht_teleop", local_dir="data/pusht_teleop", fps=10, save_videos=True, coarsity=None, policy=None, preprocessor=None, postprocessor=None):
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.fps = fps
        self.save_videos = save_videos
        self.coarsity = coarsity
        self.action_type = action_type
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        # Import and create PushT environment
        try:
            import gym_pusht
            self.env = gym.make(
                "gym_pusht/PushT-v0",
                obs_type="environment_state_agent_pos_pixels",
                render_mode="rgb_array",
                visualization_width=512,
                visualization_height=512,
                observation_width=512,
                observation_height=512,
                coarsity=self.coarsity,
                action_type=self.action_type
            )
        except ImportError:
            print("Error: gym_pusht not installed. Install with: pip install gym-pusht")
            raise

        self.current_episode = 0
        self.task_description = "Push the T-shape block to the target zone"

        # Initialize mouse input
        self.window_name = "PushT Environment - Mouse Control"
        self.mouse = MouseInput(self.window_name)
        
        self.rnd_window_name = "RND Score Over Time"
        self.rnd_scores = []
        self.max_episode_steps = 300

        # Initialize LeRobot dataset
        self.dataset = None
        self._init_dataset()

        print("\n" + "="*60)
        print("🎮 LeRobot PushT Teleoperation Recorder (Mouse Control)")
        print("="*60)
        print(f"📁 Dataset location: {self.local_dir}")
        print(f"🎬 Save videos: {save_videos}")
        print(f"📊 FPS: {fps}")
        print("="*60)
        print("🖱️  Controls:")
        print("   - Move mouse to control agent position")
        print("   - LEFT CLICK to start recording episode")
        print("   - RIGHT CLICK to finish current episode")
        print("   - MIDDLE CLICK or 'Q' key to quit")
        print("="*60)

    def _init_dataset(self):
        """Initialize LeRobot dataset with proper configuration"""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            # Define features for the dataset
            features = {
                "observation.image": {
                    "dtype": "video",
                    "shape": (3, 512, 512),
                    "names": ["channel", "height", "width"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (2,),
                    "names": ["x", "y"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (2,),
                    "names": ["x", "y"],
                },
                "observation.environment_state": {
                    "dtype": "float32",
                    "shape": (16,),
                    "names": [str(i) for i in range(16)],
                },
                "observation.block_pos": {
                    "dtype": "float32",
                    "shape": (2,),
                    "names": ["x", "y"],
                },
            }

            # Create dataset
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                root=str(self.local_dir),
                fps=self.fps,
                robot_type="pusht",
                features=features,
            )
            print(f"✅ Initialized LeRobot dataset")

        except ImportError:
            raise ImportError("LeRobotDataset not found. Install lerobot package to use dataset features.")

    def render_window(self, img, step, reward, total_reward, max_reward, action, recording=False, success=None):
        """Render environment in OpenCV window with info overlay"""
        # Create a copy to draw on
        display_img = img.copy()
        h, w = display_img.shape[:2]

        # Update mouse with window size for proper coordinate scaling
        self.mouse.set_window_size(w, h)

        # THE FOLLOWING IS NOT YET SUPPORTED FOR RELATIVE ACTIONS
        # Draw crosshair at action position (scale from env coords to window coords)
        # action_x = int((action[0] / 512.0) * w)
        # action_y = int((action[1] / 512.0) * h)
        # cv2.drawMarker(display_img, (action_x, action_y), (0, 255, 0), 
        #               cv2.MARKER_CROSS, 20, 2)

        # Add info overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Status indicator
        status = "🔴 RECORDING" if recording else "⚪ READY"
        color = (0, 0, 255) if recording else (255, 255, 255)

        # Text with background
        texts = [
            f"Episode: {self.current_episode} | Step: {step} | {status}",
            f"Reward: {reward:.2f} | Total: {total_reward:.2f} | Max: {max_reward:.2f}",
            f"Action: [{action[0]:.0f}, {action[1]:.0f}]",
        ]

        # Add success indicator if episode just finished
        if success is not None:
            success_text = "✅ SUCCESS!" if success else "❌ FAILED"
            success_color = (0, 255, 0) if success else (0, 0, 255)
            texts.append(success_text)

        y_offset = 30
        for i, text in enumerate(texts):
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(display_img, (5, y_offset - 20), 
                         (text_size[0] + 15, y_offset + 5), (0, 0, 0), -1)

            # Use special color for success/failure text
            if i == len(texts) - 1 and success is not None:
                cv2.putText(display_img, text, (10, y_offset), 
                           font, font_scale, success_color, thickness)
            else:
                cv2.putText(display_img, text, (10, y_offset), 
                           font, font_scale, color, thickness)
            y_offset += 35

        # Instructions at bottom
        instructions = [
            "LEFT CLICK: Start | RIGHT CLICK: Finish | MIDDLE/Q: Quit"
        ]
        y_offset = h - 20
        for text in instructions:
            text_size = cv2.getTextSize(text, font, font_scale - 0.1, 1)[0]
            cv2.rectangle(display_img, (5, y_offset - 20), 
                         (text_size[0] + 15, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(display_img, text, (10, y_offset), 
                       font, font_scale - 0.1, (255, 255, 255), 1)
            y_offset -= 25

        cv2.imshow(self.window_name, cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
        
    def render_rnd_graph(self):
        """Render RND score graph in separate window"""
        graph_width = 800
        graph_height = 400
        padding = 60
        
        # Create blank canvas
        canvas = np.ones((graph_height, graph_width, 3), dtype=np.uint8) * 255
        
        if len(self.rnd_scores) == 0:
            cv2.imshow(self.rnd_window_name, canvas)
            return
        
        # Calculate statistics
        current_score = self.rnd_scores[-1]
        min_score = min(self.rnd_scores)
        max_score = max(self.rnd_scores)
        
        # Handle case where all scores are the same
        score_range = max_score - min_score
        if score_range < 1e-10:
            score_range = max_score * 0.1 if max_score > 0 else 1e-6
        
        # Draw axes
        cv2.line(canvas, (padding, graph_height - padding), 
                (graph_width - padding, graph_height - padding), (0, 0, 0), 2)
        cv2.line(canvas, (padding, padding), 
                (padding, graph_height - padding), (0, 0, 0), 2)
        
        # Plot area dimensions
        plot_width = graph_width - 2 * padding
        plot_height = graph_height - 2 * padding
        
        # Draw grid lines
        for i in range(5):
            y = padding + (plot_height // 4) * i
            cv2.line(canvas, (padding, y), (graph_width - padding, y), (200, 200, 200), 1)
        
        # Plot points
        if len(self.rnd_scores) > 1:
            points = []
            for i, score in enumerate(self.rnd_scores):
                x = padding + int((i / self.max_episode_steps) * plot_width)
                normalized = (score - min_score) / score_range
                y = graph_height - padding - int(normalized * plot_height)
                points.append((x, y))
            
            # Draw line connecting points
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], (255, 0, 0), 2)
            
            # Draw current point
            cv2.circle(canvas, points[-1], 5, (0, 0, 255), -1)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Current, min, max scores (in scientific notation)
        texts = [
            f"Current: {current_score:.2e}",
            f"Min: {min_score:.2e}",
            f"Max: {max_score:.2e}",
            f"Step: {len(self.rnd_scores)}/{self.max_episode_steps}"
        ]
        
        y_offset = 25
        for text in texts:
            cv2.putText(canvas, text, (graph_width - 200, y_offset), 
                    font, font_scale, (0, 0, 0), thickness)
            y_offset += 25
        
        # Y-axis label
        cv2.putText(canvas, "RND Score", (5, 30), font, 0.6, (0, 0, 0), thickness)
        # X-axis label
        cv2.putText(canvas, "Steps", (graph_width // 2 - 30, graph_height - 10), 
                font, 0.6, (0, 0, 0), thickness)
        
        cv2.imshow(self.rnd_window_name, canvas)

    def record_episode_lerobot(self):
        """Record episode using LeRobot dataset format with mouse control"""
        print(f"\n{'='*60}")
        print(f"📹 Recording Episode {self.current_episode}")
        print(f"{'='*60}")
        print("🖱️  Move mouse over window and LEFT CLICK to start recording...")

        self.mouse.reset_flags()
        
        self.rnd_scores = []
        cv2.namedWindow(self.rnd_window_name)

        obs, info = self.env.reset()

        # Get initial rendering to set up window size
        img = self.env.render()
        h, w = img.shape[:2]
        self.mouse.set_window_size(w, h)

        # Wait for user to click to start
        while not self.mouse.recording_started:
            img = self.env.render()
            action = self.mouse.get_action()
            if self.action_type == "relative":
                action = action - self.env.unwrapped.agent.position
            self.render_window(img, 0, 0, 0, 0, action, recording=False)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or self.mouse.quit_recording:
                print("\n❌ Recording cancelled by user")
                return False

        print("🔴 Recording started!\n")

        done = False
        truncated = False
        step = 0
        total_reward = 0
        max_reward = 0

        video_writer = None
        if self.save_videos:
            video_path = self.local_dir / "videos" / f"episode_{self.current_episode:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        frame_duration = 1.0 / self.fps

        while not (done or truncated):
            loop_start = time.time()

            # Get action from mouse position
            action = self.mouse.get_action()
            if self.action_type == "relative":
                action = action - self.env.unwrapped.agent.position

            if self.policy:
                observation = preprocess_observation(obs)
                observation = self.preprocessor(observation)
                _ = self.policy.populate_queues(observation) 
                rnd_score = self.policy.predict_rnd(observation)
                self.rnd_scores.append(float(rnd_score))

                self.render_rnd_graph()

            # Step environment
            next_obs, reward, is_done, is_truncated, info = self.env.step(np.copy(action))
            total_reward += reward
            max_reward = max(max_reward, reward)

            # Render
            _ = self.env.render()

            img = obs["pixels"]

            # Initialize video writer on first frame
            if self.save_videos and video_writer is None:
                h, w = img.shape[0], img.shape[1]
                video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))

            if video_writer:
                video_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(video_frame)

            # Prepare frame for dataset
            assert img.shape == (512, 512, 3), f"Image shape is {img.shape}, expected (512, 512, 3)"

            frame = {
                "observation.environment_state": torch.from_numpy(
                    obs["environment_state"].astype(np.float32)
                ),
                "observation.block_pos": torch.from_numpy(
                    obs["block_pos"].astype(np.float32)
                ),
                "observation.image": torch.from_numpy(
                    np.array(Image.fromarray(img)).transpose(2, 0, 1)
                ),
                "observation.state": torch.from_numpy(
                    obs["agent_pos"].astype(np.float32)
                ),
                "action": torch.from_numpy(action.astype(np.float32)),
                "task": self.task_description,
            }

            self.dataset.add_frame(frame)

            obs = next_obs
            step += 1

            # Render window
            self.render_window(img, step, reward, total_reward, max_reward, action, recording=True)

            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.mouse.quit_recording:
                print("\n❌ Recording cancelled by user")
                if video_writer: 
                    video_writer.release()
                return False

            if self.mouse.finish_episode:
                print("\n✅ Episode finished by user")
                done = True

            # Check environment done
            if not done:
                done = is_done or is_truncated

            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)

        if video_writer:
            video_writer.release()

        # Display success status for X seconds
        success_display_time = 0.25
        success_start = time.time()
        while time.time() - success_start < success_display_time:
            img = self.env.render()
            self.render_window(
                img,
                step,
                reward,
                total_reward,
                max_reward,
                action,
                recording=False,
                success=info["is_success"],
            )
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

        self.dataset.save_episode()

        print(f"\n{'='*60}")
        print(f"✅ Episode {self.current_episode} saved!")
        print(f"   Steps: {step}")
        print(f"   Total Reward: {total_reward:.3f}")
        print(f"   Max Reward: {max_reward:.3f}")
        print(f"   Success: {'✅ YES' if info['is_success'] else '❌ NO'}")
        if self.save_videos:
            print(f"   Video: {video_path}")
        print(f"{'='*60}")

        return True

    def run(self, num_episodes=10):
        """Record multiple episodes"""
        print(f"\n🎯 Target: {num_episodes} episodes")

        # Create window and set up mouse callback
        cv2.namedWindow(self.window_name)
        self.mouse.setup()

        for i in range(num_episodes):
            success = self.record_episode_lerobot()
            if not success:
                print("\n⛔ Recording stopped by user")
                break
            self.current_episode += 1
            print(f"\n📊 Progress: {self.current_episode}/{num_episodes} episodes completed\n")

        if self.dataset:
            print("Finalizing dataset...")
            self.dataset.finalize()
            print("\n✅ Dataset finalized!")

        self.cleanup()
        print(f"\n{'='*60}")
        print(f"🎉 Recording Complete!")
        print(f"📁 Location: {self.local_dir}")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'env'):
            self.env.close()
        cv2.destroyAllWindows()
        try:
            cv2.destroyWindow(self.rnd_window_name)
        except:
            pass
        
def parse_custom_args():
    import argparse
    parser = argparse.ArgumentParser(description='Record PushT teleoperation episodes')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to record')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--no-video', action='store_true', help='Disable video recording')
    parser.add_argument('--action_type', type=str, help='relative or absolute actions')
    parser.add_argument('--coarsity', type=str, help='Coarsity of PushT environment')
    parser.add_argument('--dir_prefix', type=str, default="TEST", help='Prefix for directory name')
    parser.add_argument('--rnd_network_path', type=str, default=None, help='Path to trained rnd network') # "outputs/train/2025-12-15/15-09-16_pusht_diffusion_fine_l1_RND_OOD/rnd/rnd.pth"
    
    # Parse only known args to avoid conflicts with parser.wrap()
    args, unknown = parser.parse_known_args()
    return args

@parser.wrap()
def main(cfg: EvalPipelineConfig):
    # Parse your custom arguments
    args = parse_custom_args()
    
    # this requires that the correct config (that would also be passed to leroboto_eval) is passed
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    
    if args.rnd_network_path:
        rnd = RND(config = None)
        rnd.load_state_dict(torch.load(args.rnd_network_path))
        rnd.eval()
        rnd.cuda()
        policy.rnd = rnd
    
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    
    
    # Append current date and time to directory name
    now = time.localtime()
    dir_suffix = f"Y{now.tm_year:04d}_M{now.tm_mon:02d}_D{now.tm_mday:02d}_H{now.tm_hour:02d}_M{now.tm_min:02d}_S{now.tm_sec:02d}"
    
    repo_id = f"{args.dir_prefix}_pusht_teleop_{dir_suffix}_{args.coarsity}_{args.action_type}"
    dir = f"pusht_teleop_data/{repo_id}"
    
    
    
    recorder = PushTTeleopRecorder(
        repo_id=repo_id,
        local_dir=dir,
        fps=args.fps,
        save_videos=not args.no_video,
        coarsity=args.coarsity,
        action_type=args.action_type,
        policy=policy if args.rnd_network_path else None,
        preprocessor=preprocessor if args.rnd_network_path else None,
        postprocessor=postprocessor if args.rnd_network_path else None,
    )
    recorder.run(num_episodes=args.episodes)


if __name__ == "__main__":
    main()
