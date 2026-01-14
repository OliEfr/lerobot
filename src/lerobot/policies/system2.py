from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from scipy import ndimage
import numpy as np
from lerobot.utils.constants import first_person_gripper_mask, LIBERO_GRIPPER_THRESHOLD


# ==================== Success Criterions ====================

@runtime_checkable
class SuccessCriterion(Protocol):
    """Protocol for success criteria - defines the interface."""

    def __call__(
        self,
        observation: dict,
        segmented_frames: dict[str, np.ndarray] | None = None,
    ) -> bool:
        """
        Check if success criterion is met.

        Args:
            observation: Current observation dict (contains all sensor data)
            segmented_frames: SAM3 segmentation masks by camera name (optional)

        Returns:
            True if criterion is met, False otherwise
        """
        ...


@dataclass(frozen=True)
class DepthThreshold(SuccessCriterion):
    """Success when object depth is below threshold."""

    threshold: float
    depth_key: str  # observation key for depth (e.g., "observation.depths.depth2")
    camera: str = "1st_person"  # camera name in segmented_frames dict

    def __call__(
        self,
        observation: dict,
        segmented_frames: dict[str, np.ndarray] | None = None,
    ) -> bool:
        if segmented_frames is None:
            return False

        if self.camera not in segmented_frames:
            return False

        # Extract depth from observation using configured key
        if self.depth_key not in observation:
            return False

        depth_obs = observation[self.depth_key][0, ..., 0].cpu().numpy()

        # Apply mask and compute min depth
        mask = segmented_frames[self.camera] > 0
        # Erode (shrink) mask to avoid edge artifacts
        mask = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), iterations=2)

        # Mask out gripper for 1st person camera
        if self.camera == "1st_person":
            mask = mask & first_person_gripper_mask  # Exclude gripper pixels

        object_depths = depth_obs[mask]

        if len(object_depths) == 0:
            return False

        return object_depths.min() < self.threshold


@dataclass(frozen=True)
class DepthThresholdAbove(SuccessCriterion):
    """Success when object depth is above threshold (farther away)."""

    threshold: float
    depth_key: str  # observation key for depth (e.g., "observation.depths.depth2")
    camera: str = "1st_person"  # camera name in segmented_frames dict

    def __call__(
        self,
        observation: dict,
        segmented_frames: dict[str, np.ndarray] | None = None,
    ) -> bool:
        if segmented_frames is None:
            return False

        if self.camera not in segmented_frames:
            return False

        # Extract depth from observation using configured key
        if self.depth_key not in observation:
            return False

        depth_obs = observation[self.depth_key][0, ..., 0].cpu().numpy()

        mask = segmented_frames[self.camera] > 0
        # Erode (shrink) mask to avoid edge artifacts
        mask = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), iterations=2)

        # Mask out gripper for 1st person camera
        if self.camera == "1st_person":
            mask = mask & first_person_gripper_mask  # Exclude gripper pixels

        object_depths = depth_obs[mask]

        if len(object_depths) == 0:
            return False

        return object_depths.min() > self.threshold


@dataclass(frozen=True)
class NeverSucceeds(SuccessCriterion):
    """Never succeeds - stage runs until episode ends."""

    def __call__(
        self,
        observation: dict,  # noqa: ARG002
        segmented_frames: dict[str, np.ndarray] | None = None,  # noqa: ARG002
    ) -> bool:
        return False


@dataclass
class GripperClosedForN(SuccessCriterion):
    """Success when gripper has been closed for N consecutive timesteps."""

    threshold: float = LIBERO_GRIPPER_THRESHOLD  # gripper joint position threshold for "closed"
    required_steps: int = 10  # number of consecutive steps required
    state_key: str = "observation.state"  # observation key for robot state
    gripper_indices: tuple[int, int] = (6, 7)  # indices of gripper joints in state

    _consecutive_count: int = field(default=0, init=False, repr=False)

    def __call__(
        self,
        observation: dict,
        segmented_frames: dict[str, np.ndarray] | None = None,  # noqa: ARG002
    ) -> bool:
        if self.state_key not in observation:
            return False

        state = observation[self.state_key][0].cpu().numpy()
        g1 = state[self.gripper_indices[0]]
        g2 = state[self.gripper_indices[1]]

        gripper_closed = (abs(g1) < self.threshold) and (abs(g2) < self.threshold)

        if gripper_closed:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        return self._consecutive_count >= self.required_steps

    def reset(self) -> None:
        """Reset consecutive count for new episode."""
        object.__setattr__(self, "_consecutive_count", 0)


@dataclass
class GripperOpenedForN(SuccessCriterion):
    """Success when gripper has been opened for N consecutive timesteps."""

    threshold: float = LIBERO_GRIPPER_THRESHOLD  # gripper joint position threshold for "opened"
    required_steps: int = 10  # number of consecutive steps required
    state_key: str = "observation.state"  # observation key for robot state
    gripper_indices: tuple[int, int] = (6, 7)  # indices of gripper joints in state

    _consecutive_count: int = field(default=0, init=False, repr=False)

    def __call__(
        self,
        observation: dict,
        segmented_frames: dict[str, np.ndarray] | None = None,  # noqa: ARG002
    ) -> bool:
        if self.state_key not in observation:
            return False

        state = observation[self.state_key][0].cpu().numpy()
        g1 = state[self.gripper_indices[0]]
        g2 = state[self.gripper_indices[1]]

        # Gripper is opened when both joints are ABOVE threshold (using absolute values)
        gripper_opened = (abs(g1) > self.threshold) and (abs(g2) > self.threshold)

        if gripper_opened:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        return self._consecutive_count >= self.required_steps

    def reset(self) -> None:
        """Reset consecutive count for new episode."""
        object.__setattr__(self, "_consecutive_count", 0)


@dataclass
class TimestepCounter(SuccessCriterion):
    """Success after N timesteps have elapsed."""

    required_steps: int = 50  # number of timesteps required
    _step_count: int = field(default=0, init=False, repr=False)

    def __call__(
        self,
        observation: dict,  # noqa: ARG002
        segmented_frames: dict[str, np.ndarray] | None = None,  # noqa: ARG002
    ) -> bool:
        self._step_count += 1
        return self._step_count >= self.required_steps

    def reset(self) -> None:
        """Reset step count for new episode."""
        object.__setattr__(self, "_step_count", 0)


@dataclass
class AndCriterion(SuccessCriterion):
    """Success when BOTH criteria are met simultaneously."""

    criterion_a: SuccessCriterion
    criterion_b: SuccessCriterion

    def __call__(
        self,
        observation: dict,
        segmented_frames: dict[str, np.ndarray] | None = None,
    ) -> bool:
        # Evaluate both criteria to ensure state updates (e.g., gripper counter)
        result_a = self.criterion_a(observation, segmented_frames)
        result_b = self.criterion_b(observation, segmented_frames)
        return result_a and result_b

    def reset(self) -> None:
        """Reset both criteria if they support it."""
        if hasattr(self.criterion_a, "reset"):
            self.criterion_a.reset()
        if hasattr(self.criterion_b, "reset"):
            self.criterion_b.reset()


# ==================== Stage & System2 ====================

@dataclass
class Stage:
    """Configuration for a single stage."""

    name: str  # human-readable name for the stage
    mode: Literal["sam3", "policy", "home"]
    success_criterion: SuccessCriterion
    sam3_prompt: str | None = None  # model prompt for this stage
    vs_camera: Literal["1st_person", "3rd_person"] | None = None  # camera used for visual servoing
    sam3_stage: int = 0  # SAM3 stage counter - increment for each sam3 stage to trigger reset


# Import after class definitions to avoid circular imports
from lerobot.utils.constants import VISUAL_SERVOING_SETTINGS


class System2:
    """ State manager """
    def __init__(self):
        # Hardcoded stages - replicate current behavior for testing
        self.stages = [
            Stage(
                name="approach_mug",
                mode="sam3",
                success_criterion=DepthThreshold(
                    threshold=0.15,
                    depth_key=VISUAL_SERVOING_SETTINGS["1st_person"]["depth_name"],
                    camera="1st_person",
                ),
                sam3_prompt="yellow white mug",
                vs_camera="3rd_person",
                sam3_stage=0,
            ),
            Stage(
                name="grasp_mug",
                mode="policy",
                success_criterion=GripperClosedForN(
                        threshold=0.02,
                        required_steps=10,
                        state_key="observation.state",
                        gripper_indices=(6, 7),
                    ),
                sam3_prompt="yellow white mug",
                vs_camera=None,
                sam3_stage=0,
            ),
            Stage(
                name="return_home",
                mode="home",
                success_criterion=TimestepCounter(required_steps=50),
                sam3_prompt=None,
                vs_camera=None,
                sam3_stage=0,  # Same as grasp_mug since no SAM3 reset needed
            ),
            Stage(
                name="move_to_plate",
                mode="sam3",
                success_criterion=DepthThreshold(
                    threshold=0.15,
                    depth_key=VISUAL_SERVOING_SETTINGS["1st_person"]["depth_name"],
                    camera="1st_person",
                ),
                sam3_prompt="left plate",
                vs_camera="3rd_person",
                sam3_stage=1,
            ),
            Stage(
                name="place_on_plate",
                mode="policy",
                success_criterion=AndCriterion(
                    criterion_a=DepthThresholdAbove(
                        threshold=0.15,
                        depth_key=VISUAL_SERVOING_SETTINGS["1st_person"]["depth_name"],
                        camera="1st_person",
                    ),
                    criterion_b=GripperOpenedForN(
                        threshold=0.02,
                        required_steps=10,
                        state_key="observation.state",
                        gripper_indices=(6, 7),
                    ),
                ),
                sam3_prompt="plate",
                vs_camera=None,
                sam3_stage=2,
            ),
            Stage(
                name="return_home",
                mode="home",
                success_criterion=TimestepCounter(required_steps=220),
                sam3_prompt=None,
                vs_camera=None,
                sam3_stage=2,  # Same as grasp_mug since no SAM3 reset needed
            ),
            Stage(
                name="approach_mug2",
                mode="sam3",
                success_criterion=DepthThreshold(
                    threshold=0.2,
                    depth_key=VISUAL_SERVOING_SETTINGS["1st_person"]["depth_name"],
                    camera="1st_person",
                ),
                sam3_prompt="grey mug on the right",
                vs_camera="3rd_person",
                sam3_stage=3,
            ),
            Stage(
                name="grasp_mug2",
                mode="policy",
                success_criterion=GripperClosedForN(
                        threshold=0.02,
                        required_steps=10,
                        state_key="observation.state",
                        gripper_indices=(6, 7),
                    ),
                sam3_prompt="grey mug on the right",
                vs_camera=None,
                sam3_stage=3,
            ),
            Stage(
                name="return_home",
                mode="home",
                success_criterion=TimestepCounter(required_steps=50),
                sam3_prompt=None,
                vs_camera=None,
                sam3_stage=3,
            ),
            Stage(
                name="move_to_plate2",
                mode="sam3",
                success_criterion=DepthThreshold(
                    threshold=0.2,
                    depth_key=VISUAL_SERVOING_SETTINGS["1st_person"]["depth_name"],
                    camera="1st_person",
                ),
                sam3_prompt="left plate",
                vs_camera="3rd_person",
                sam3_stage=4,
            ),
            Stage(
                name="place_on_plate2",
                mode="policy",
                success_criterion=AndCriterion(
                    criterion_a=DepthThresholdAbove(
                        threshold=0.2,
                        depth_key=VISUAL_SERVOING_SETTINGS["1st_person"]["depth_name"],
                        camera="1st_person",
                    ),
                    criterion_b=GripperOpenedForN(
                        threshold=0.02,
                        required_steps=10,
                        state_key="observation.state",
                        gripper_indices=(6, 7),
                    ),
                ),
                sam3_prompt="plate",
                vs_camera=None,
                sam3_stage=4,
            ),
            Stage(
                name="return_home",
                mode="home",
                success_criterion=TimestepCounter(required_steps=50),
                sam3_prompt=None,
                vs_camera=None,
                sam3_stage=4,
            ),
        ]
        self.current_stage_index = 0

    def get_current_stage(self) -> Stage:
        return self.stages[self.current_stage_index]

    def get_sam3_stage(self) -> int:
        return self.get_current_stage().sam3_stage

    def check_and_advance(
        self,
        observation: dict,
        segmented_frames: dict[str, np.ndarray] | None = None,
    ) -> bool:
        """
        Check if current stage's success criterion is met and advance if so.

        Args:
            observation: Current observation dict (contains all sensor data)
            segmented_frames: SAM3 segmentation masks by camera name (optional)

        Returns:
            True if stage was advanced, False otherwise
        """
        # Don't advance past last stage
        if self.current_stage_index >= len(self.stages):
            print("[System2] All stages completed.")
            return False

        current_stage = self.get_current_stage()

        is_met = current_stage.success_criterion(
            observation=observation,
            segmented_frames=segmented_frames,
        )

        if is_met:
            next_stage_index = self.current_stage_index + 1
            if next_stage_index < len(self.stages):
                next_stage_name = self.stages[next_stage_index].name
                print(f"[System2] Stage completed: '{current_stage.name}' -> advancing to stage '{next_stage_name}'")
            self.current_stage_index += 1
            return True

        return False

    def reset(self) -> None:
        """Reset to first stage for new episode."""
        self.current_stage_index = 0
        # Reset any stateful success criteria
        for stage in self.stages:
            if hasattr(stage.success_criterion, "reset"):
                stage.success_criterion.reset()

    def has_sam3_stage(self) -> bool:
        """Check if any stage uses SAM3 mode."""
        return any(stage.mode == "sam3" for stage in self.stages)
