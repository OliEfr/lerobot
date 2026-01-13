from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from lerobot.utils.constants import first_person_gripper_mask


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

        # Mask out gripper for 1st person camera
        if self.camera == "1st_person":
            mask = mask & first_person_gripper_mask  # Exclude gripper pixels

        object_depths = depth_obs[mask]

        if len(object_depths) == 0:
            return False

        return object_depths.min() < self.threshold


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

    threshold: float = 0.02  # gripper joint position threshold for "closed"
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

        gripper_closed = (g1 < self.threshold) and (g2 < self.threshold)

        if gripper_closed:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        return self._consecutive_count >= self.required_steps

    def reset(self) -> None:
        """Reset consecutive count for new episode."""
        object.__setattr__(self, "_consecutive_count", 0)


# ==================== Stage & System2 ====================

@dataclass
class Stage:
    """Configuration for a single stage."""

    name: str  # human-readable name for the stage
    mode: Literal["sam3", "policy"]
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
                success_criterion=GripperClosedForN(),
                sam3_prompt="yellow white mug",
                vs_camera=None,
                sam3_stage=0,
            ),
            Stage(
                name="place_on_plate",
                mode="sam3",
                success_criterion=NeverSucceeds(),
                sam3_prompt="plate",
                vs_camera="3rd_person",
                sam3_stage=1,
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
            return False

        current_stage = self.get_current_stage()

        is_met = current_stage.success_criterion(
            observation=observation,
            segmented_frames=segmented_frames,
        )

        if is_met:
            print(f"[System2] Stage completed: '{current_stage.name}' -> advancing to stage {self.current_stage_index + 1}")
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
