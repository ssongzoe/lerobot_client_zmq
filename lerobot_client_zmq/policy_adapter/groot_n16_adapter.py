"""GR00T N1.6 observation/action adapter for the existing ZMQ server schema.

Only translates the robot's observation and the server's predicted actions.
Networking, action queue management, and dispatch live outside this module.
No action clipping, smoothing, or interpolation is applied here.
"""

from typing import Any
import numpy as np

from .base import ACTION_NAMES, BasePolicyAdapter


# The original client uses right arm, left arm, right gripper, left gripper
# internally, even though the GR00T server sends its parts under separate keys.
RIGHT_ARM_KEYS = ACTION_NAMES[:7]
LEFT_ARM_KEYS = ACTION_NAMES[7:14]
RIGHT_GRIPPER_KEY = ACTION_NAMES[14]
LEFT_GRIPPER_KEY = ACTION_NAMES[15]


def _ensure_uint8_hwc_image(image: Any) -> np.ndarray:
    """Preserve the original GR00T requirement: uint8 image in HWC layout."""
    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[-1] != 3:
        raise ValueError(f"Expected image with shape (H, W, 3), got {image_arr.shape}")
    if image_arr.dtype != np.uint8:
        raise ValueError(f"Expected image dtype uint8, got {image_arr.dtype}")
    return np.ascontiguousarray(image_arr)


class GrootN16Adapter(BasePolicyAdapter):
    """Map RB-Y1 observations and GR00T N1.6 actions without changing values."""

    name = "groot_n16"

    def build_observation(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        # Local `.pos` feature keys are normalized by the base adapter;
        # the payload sent to the GR00T server stays exactly as before.
        observation = self.normalize_observation(raw_observation)
        required_keys = [
            *ACTION_NAMES,
            self.front_camera_key,
            self.left_wrist_camera_key,
            self.right_wrist_camera_key,
        ]
        missing_keys = [key for key in required_keys if key not in observation]
        if missing_keys:
            raise KeyError(f"Raw observation missing keys required by GR00T backend: {missing_keys}")

        front = _ensure_uint8_hwc_image(observation[self.front_camera_key])
        left = _ensure_uint8_hwc_image(observation[self.left_wrist_camera_key])
        right = _ensure_uint8_hwc_image(observation[self.right_wrist_camera_key])

        right_arm = np.asarray([observation[k] for k in RIGHT_ARM_KEYS], dtype=np.float32)
        left_arm = np.asarray([observation[k] for k in LEFT_ARM_KEYS], dtype=np.float32)
        right_gripper = np.asarray([observation[RIGHT_GRIPPER_KEY]], dtype=np.float32)
        left_gripper = np.asarray([observation[LEFT_GRIPPER_KEY]], dtype=np.float32)
        task = str(observation.get("task", ""))

        return {
            "video": {
                "cam_front_head": front[np.newaxis, np.newaxis],
                "cam_left_wrist": left[np.newaxis, np.newaxis],
                "cam_right_wrist": right[np.newaxis, np.newaxis],
            },
            "state": {
                "left_arm": left_arm[np.newaxis, np.newaxis],
                "right_arm": right_arm[np.newaxis, np.newaxis],
                "left_gripper": left_gripper[np.newaxis, np.newaxis],
                "right_gripper": right_gripper[np.newaxis, np.newaxis],
            },
            "language": {
                "annotation.human.task_description": [[task]],
            },
        }

    def parse_actions(self, action_response: dict[str, Any]) -> np.ndarray:
        """Convert GR00T's (1,T,7/1) parts into (T,16), without altering values."""
        left_arm = np.asarray(action_response.get("left_arm"), dtype=np.float32)
        right_arm = np.asarray(action_response.get("right_arm"), dtype=np.float32)
        left_gripper = np.asarray(action_response.get("left_gripper"), dtype=np.float32)
        right_gripper = np.asarray(action_response.get("right_gripper"), dtype=np.float32)

        for key, values, dof in (
            ("left_arm", left_arm, 7),
            ("right_arm", right_arm, 7),
            ("left_gripper", left_gripper, 1),
            ("right_gripper", right_gripper, 1),
        ):
            if values.shape[:1] != (1,) or values.ndim != 3 or values.shape[2] != dof:
                raise ValueError(f"Expected {key} shape (1, T, {dof}), got {values.shape}")

        lengths = (left_arm.shape[1], right_arm.shape[1], left_gripper.shape[1], right_gripper.shape[1])
        if len(set(lengths)) != 1:
            raise ValueError(
                "GR00T action chunk lengths do not match: "
                f"left_arm={lengths[0]}, right_arm={lengths[1]}, "
                f"left_gripper={lengths[2]}, right_gripper={lengths[3]}"
            )

        # Exactly the ordering of the 0.6.0 client, including both grippers.
        return np.concatenate(
            (right_arm[0], left_arm[0], right_gripper[0], left_gripper[0]),
            axis=-1,
        )
