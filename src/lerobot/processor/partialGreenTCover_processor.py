#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import (
    COVER_GREEN_T_END_X_STATE,
    COVER_GREEN_T_END_Y_STATE,
    COVER_GREEN_T_START_X_STATE,
    COVER_GREEN_T_START_Y_STATE,
)

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="partial_green_t_cover_processor")
class PartialGreenTCoverProcessorStep(ObservationProcessorStep):
    """This processor decides whether to return the environment_state (keypoints of the tee) or zeros depending on some condition."""

    threshold: int = 6
    _metrics: dict = field(default_factory=dict, init=False)

    def observation(self, observation):

        assert observation["observation.environment_state"].shape[-1] == 16, "Expected 16 features in environment_state of pusht."

        # Add history dimension if not present: ensure shape is [batch_size, history_steps, 16]
        if observation["observation.environment_state"].ndim == 2:
            observation["observation.environment_state"] = observation["observation.environment_state"].unsqueeze(1)  # [batch_size, 16] -> [batch_size, 1, 16]
            squeeze_output = True
        elif observation["observation.environment_state"].ndim == 3:
            squeeze_output = False
        else:
            raise ValueError("Expected 2 or 3 dimensions in environment_state of pusht.")


        keypoints_reshaped = observation["observation.environment_state"].view(
            observation["observation.environment_state"].shape[0],
            observation["observation.environment_state"].shape[1],
            -1,
            2,
        )

        x_coords = keypoints_reshaped[..., 0]
        y_coords = keypoints_reshaped[..., 1]

        inside_x = (x_coords >= COVER_GREEN_T_START_X_STATE) & (
            x_coords <= COVER_GREEN_T_END_X_STATE
        )
        inside_y = (y_coords >= COVER_GREEN_T_START_Y_STATE) & (
            y_coords <= COVER_GREEN_T_END_Y_STATE
        )
        inside_rect = inside_x & inside_y

        num_covered = inside_rect.sum(dim=2)

        sufficient_coverage = (num_covered >= self.threshold)

        self._metrics = {
            "partial_green_t_cover_processor/num_covered_mean": num_covered.float().mean().detach().clone().item(),
            "partial_green_t_cover_processor/num_covered_ratio": (num_covered.float().mean() / keypoints_reshaped.shape[2]).detach().clone().item(),
            "partial_green_t_cover_processor/sufficient_coverage_ratio": sufficient_coverage.float().mean().detach().clone().item()
        }

        sufficient_coverage = sufficient_coverage.unsqueeze(2)

        observation["observation.environment_state"] = observation["observation.environment_state"] * sufficient_coverage

        # Remove history dimension if it was added
        if squeeze_output:
            observation["observation.environment_state"] = observation["observation.environment_state"].squeeze(1)

        return observation

    def transform_features(self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_metrics(self):
        """Returns the current metrics."""
        return self._metrics
