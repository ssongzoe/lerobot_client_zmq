"""Common interface for Pi0.5, GR00T N1.6, and future ZMQ policy adapters.

The policy server's observation schema belongs to each concrete adapter.
The client-side action order stays the same as the legacy 0.6.0 client:
right arm (7), left arm (7), right gripper (1), left gripper (1).
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch

from ..helpers import RawObservation, TimedAction


ACTION_NAMES = (
    *(f"right_arm_{i}" for i in range(7)),
    *(f"left_arm_{i}" for i in range(7)),
    "right_gripper_0",
    "left_gripper_0",
)


def _find_feature_key(features: Any, name: str) -> str:
    """Map a canonical joint name to its LeRobot robot-driver feature key.

    Some 0.7.0 robot drivers use `.pos` suffixes. This only changes the
    local robot API key; it never changes the server payload or action order.
    """
    for key in (name, f"{name}.pos"):
        if key in features:
            return key
    raise KeyError(f"Missing robot feature: {name} (also checked {name}.pos)")


class BasePolicyAdapter(ABC):
    """Translate between the robot's features and a model's ZMQ schema.

    Subclasses implement only the model-specific payload and response shapes.
    ZMQ sockets, action queues, dispatch, and robot lifecycle live elsewhere.
    """

    name: str = "base"

    def __init__(
        self,
        front_camera_key: str = "front",
        left_wrist_camera_key: str = "left",
        right_wrist_camera_key: str = "right",
    ) -> None:
        self.front_camera_key = front_camera_key
        self.left_wrist_camera_key = left_wrist_camera_key
        self.right_wrist_camera_key = right_wrist_camera_key
        self._action_keys: dict[str, str] = {}
        self._observation_keys: dict[str, str] = {}

    def validate_robot(self, robot: Any) -> None:
        """Check and cache the robot's 16 action/state feature mappings.

        Supports both legacy names and 0.7.0-style `.pos` names without
        changing the 16-dimensional policy-server schema.
        """
        self._action_keys = {
            name: _find_feature_key(robot.action_features, name)
            for name in ACTION_NAMES
        }
        self._observation_keys = {
            name: _find_feature_key(robot.observation_features, name)
            for name in ACTION_NAMES
        }
        for camera_key in (
            self.front_camera_key,
            self.left_wrist_camera_key,
            self.right_wrist_camera_key,
        ):
            if camera_key not in robot.observation_features:
                raise KeyError(f"Missing robot camera feature: {camera_key}")

    def normalize_observation(self, raw_observation: RawObservation) -> RawObservation:
        """Expose canonical joint keys without modifying sensor values.

        Concrete adapters call this before building their model-specific
        observation dict. Camera arrays are kept unchanged here.
        """
        observation = dict(raw_observation)
        for name, robot_key in self._observation_keys.items():
            observation[name] = raw_observation[robot_key]
        return observation

    @abstractmethod
    def build_observation(self, raw_observation: RawObservation) -> dict[str, Any]:
        """Build the exact observation payload expected by this ZMQ server."""

    @abstractmethod
    def parse_actions(self, action_response: dict[str, Any]) -> np.ndarray:
        """Return server actions as (T, 16) in the legacy joint order.

        Only model-specific reshaping/reordering belongs here; no smoothing,
        clamping, interpolation, or other action-value modification.
        """

    def convert_actions(
        self,
        action_response: dict[str, Any],
        *,
        timestamp: float,
        timestep: int,
        environment_dt: float,
        client_device: str = "cpu",
    ) -> list[TimedAction]:
        """Tag each predicted action with the original 0.6.0 timing rules."""
        actions = np.asarray(self.parse_actions(action_response), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(ACTION_NAMES):
            raise ValueError(f"Expected (T, {len(ACTION_NAMES)}) actions, got {actions.shape}")

        device = torch.device(client_device)
        return [
            TimedAction(
                timestamp=timestamp + i * environment_dt,
                timestep=timestep + i,
                action=torch.from_numpy(np.ascontiguousarray(action)).to(
                    device=device, dtype=torch.float32
                ),
            )
            for i, action in enumerate(actions)
        ]

    def to_robot_action(self, action: torch.Tensor | np.ndarray) -> dict[str, float]:
        """Send a 16-D policy action to the robot without changing its values.

        Only feature names are mapped. The control loop decides when to send.
        """
        if len(action) != len(ACTION_NAMES):
            raise ValueError(f"Expected {len(ACTION_NAMES)} action values, got {len(action)}")
        return {
            self._action_keys[name]: float(action[i])
            for i, name in enumerate(ACTION_NAMES)
        }
