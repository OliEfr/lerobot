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
from abc import abstractmethod
from dataclasses import dataclass

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from .core import EnvTransition, TransitionKey


from .pipeline import ObservationProcessorStep, ProcessorStepRegistry

from lerobot.utils.constants import OBS_LANGUAGE, OBS_TASK_IDS


@dataclass
@ProcessorStepRegistry.register(name="add_task_to_observation_processor")
class AddTaskToObservationProcessor(ObservationProcessorStep):
    task_key: str = "task"

    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """
        Extracts the task description(s) from the transition's complementary data.

        Args:
            transition: The environment transition.

        Returns:
            A list of task strings, or None if the task key is not found or the value is None.
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data[self.task_key]
        if task is None:
            raise ValueError("Task extracted from Complementary data is None")

        # Standardize to a list of strings
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            return task

        return None

    def observation(self, observation):

        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        # Create a new observation dict to avoid modifying the original in place
        new_observation = dict(observation)
        new_observation[OBS_LANGUAGE] = task
        new_observation[OBS_TASK_IDS] = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["task_index"]

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


