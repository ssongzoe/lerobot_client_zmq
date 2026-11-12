"""
Pi0.5 observation/action adapter for the existing ZMQ server schema.

This module only converts robot observations and policy outputs. It does not
own a socket or modify predicted action values (no clipping or smoothing).
"""
from typing import Any
import numpy as np
from .base import ACTION_NAMES, BasePolicyAdapter


# Preserve the 0.6.0 policy state/action layout:
# right arm (7), left arm (7), right gripper (1), left gripper (1).
PI05_STATE_KEYS = ACTION_NAMES
ACTION_DOF = len(ACTION_NAMES)
ARM_DOF = 7
GRIPPER_DOF = 1


def ensure_uint8_hwc_image(image: Any) -> np.ndarray:
    """Use exactly the image conversion rules of the old pi05_zmq client."""
    image_arr = np.asarray(image)

    if image_arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {image_arr.shape}")

    if image_arr.shape[0] == 3 and image_arr.shape[-1] != 3:
        image_arr = np.transpose(image_arr, (1, 2, 0))

    if image_arr.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image_arr.shape}")

    if image_arr.dtype == np.uint8:
        return np.ascontiguousarray(image_arr)

    if np.issubdtype(image_arr.dtype, np.floating):
        max_val = image_arr.max(initial=0.0)
        image_arr = np.clip(image_arr, 0.0, 1.0 if max_val <= 1.0 else 255.0)
        if max_val <= 1.0:
            image_arr = image_arr * 255.0

    return np.ascontiguousarray(image_arr.astype(np.uint8))


def _squeeze_optional_batch(array: np.ndarray) -> np.ndarray:
    """Remove an optional leading size-one batch axis as in the old client."""
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def _get_action_part(action_dict: dict[str, Any], key: str, dim: int) -> np.ndarray:
    if key not in action_dict:
        raise KeyError(f"pi05 action response missing key: {key}")

    action = _squeeze_optional_batch(np.asarray(action_dict[key], dtype=np.float32))
    if action.ndim != 2 or action.shape[1] != dim:
        raise ValueError(
            f"Expected {key} shape (T, {dim}) or (1, T, {dim}), got {action.shape}"
        )
    return action


class Pi05Adapter(BasePolicyAdapter):
    """Convert RB-Y1 observations and Pi0.5 responses without changing schema.

    Call ``validate_robot(robot)`` once before ``build_observation`` when the
    robot uses LeRobot 0.7.0 feature names (e.g. ``right_arm_0.pos``).
    """

    name = "pi05"

    def build_observation(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        # Normalize only local robot feature names; server-side keys stay the same.
        observation = self.normalize_observation(raw_observation)
        required_keys = [
            *PI05_STATE_KEYS,
            self.front_camera_key,
            self.left_wrist_camera_key,
            self.right_wrist_camera_key,
        ]
        missing_keys = [key for key in required_keys if key not in observation]
        if missing_keys:
            raise KeyError(f"Raw observation missing keys required by pi05 backend: {missing_keys}")

        state = np.asarray([observation[key] for key in PI05_STATE_KEYS], dtype=np.float32)
        task = str(observation.get("task", ""))

        return {
            "observation/front_image": ensure_uint8_hwc_image(observation[self.front_camera_key]),
            "observation/left_image": ensure_uint8_hwc_image(observation[self.left_wrist_camera_key]),
            "observation/right_image": ensure_uint8_hwc_image(observation[self.right_wrist_camera_key]),
            "observation/state": state,
            "prompt": task,
        }

    def parse_actions(self, action_response: dict[str, Any]) -> np.ndarray:
        """Return (T,16) actions, keeping the legacy right/left/gripper order.

        Supports both the combined ``actions`` response and separate arm and
        gripper fields used by the old Pi0.5 adapter.
        """
        if "actions" in action_response:
            actions = _squeeze_optional_batch(
                np.asarray(action_response["actions"], dtype=np.float32)
            )
            if actions.ndim != 2 or actions.shape[1] < ACTION_DOF:
                raise ValueError(
                    f"Expected pi05 actions shape (T, >=16) or (1, T, >=16), got {actions.shape}"
                )
            return np.ascontiguousarray(actions[:, :ACTION_DOF])

        right_arm = _get_action_part(action_response, "right_arm", ARM_DOF)
        left_arm = _get_action_part(action_response, "left_arm", ARM_DOF)
        right_gripper = _get_action_part(action_response, "right_gripper", GRIPPER_DOF)
        left_gripper = _get_action_part(action_response, "left_gripper", GRIPPER_DOF)

        lengths = [
            right_arm.shape[0],
            left_arm.shape[0],
            right_gripper.shape[0],
            left_gripper.shape[0],
        ]
        if len(set(lengths)) != 1:
            raise ValueError(f"pi05 action chunk lengths do not match: {lengths}")

        actions = np.concatenate(
            [right_arm, left_arm, right_gripper, left_gripper], axis=-1
        )
        return np.ascontiguousarray(actions.astype(np.float32))
