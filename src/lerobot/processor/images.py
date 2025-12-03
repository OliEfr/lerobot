from dataclasses import dataclass
from typing import Any

from torchvision.transforms import v2
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
)


@ProcessorStepRegistry.register("auto_image_resize_processor")
@dataclass
class AutoImageResizeProcessorStep(ObservationProcessorStep):
    """
    Resizes image observations automatically to policy input size.

    This step iterates through specified image keys in an observation dictionary and applies resizeing. 

    Attributes:
        resize_params: A dictionary of image keys with their corresponding target resize.
    """

    input_features: dict[str, Any] | None = None

    is_first_it = True

    def observation(self, observation: dict) -> dict:
        for feature in self.input_features:
            if "image" in feature:
                policy_feat_size = self.input_features[feature].shape[-2:]
                observation_size = observation[feature].shape[-2:]
                if policy_feat_size != observation_size:
                    observation[feature] = v2.functional.resize(
                        observation[feature],
                        size=policy_feat_size,
                    )

                    if self.is_first_it:
                        print(f"ALERT: Auto-Resizing {feature} from {observation_size} to {observation[feature].shape} for policy input.")

        self.is_first_it = False

        return observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary with the crop parameters and resize dimensions.
        """
        return {
            "input_features": self.input_features,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
