"""Configuration for the LeRobot 0.7.0 ZMQ-only robot client.

Keep the old client CLI field names and action-chunk settings.  LeRobot
0.7.0's parser (used later in rollout.py) handles --robot.type and plugins.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from lerobot.robots.config import RobotConfig


# Preserve the exact aggregation equations from the 0.6.0 client.
AGGREGATE_FUNCTIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the legacy same-timestep action-chunk aggregation function."""
    try:
        return AGGREGATE_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown aggregate_fn_name {name!r}; choose from {tuple(AGGREGATE_FUNCTIONS)}") from exc


# Both spellings are accepted; model-specific adapters are implemented later.
_BACKEND_ALIASES = {
    "pi05": "pi05",
    "pi05_zmq": "pi05",
    "groot_n16": "groot_n16",
    "groot_zmq": "groot_n16",
}


@dataclass
class RobotClientConfig:
    """Client settings; compatible with the LeRobot 0.7.0 RobotConfig CLI.

    Policy inference remains on the ZMQ server.  No local policy/model is
    instantiated, and no action clamping or smoothing is configured here.
    """

    robot: RobotConfig = field(metadata={"help": "LeRobot robot configuration, e.g. --robot.type=rby1"})

    # Remote ZMQ inference
    backend: str = "pi05"  # pi05 | groot_n16; also accept legacy *_zmq names
    server_address: str = "localhost:5556"
    zmq_timeout_ms: int = 30_000
    task: str = ""
    client_device: str = "cpu"

    # Preserve legacy chunk timing and aggregation.
    fps: int = 30
    actions_per_chunk: int = 50  # Legacy CLI field; old ZMQ path did not truncate responses with it.
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "weighted_average"
    guard_steps: int = 5

    # Raw robot observation camera-name mapping (not policy-server schema names).
    front_camera_key: str = "front"
    left_wrist_camera_key: str = "left"
    right_wrist_camera_key: str = "right"

    # Retained for legacy CLI compatibility; old ZMQ observation builder did not use crops.
    image_crop_params: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    debug_visualize_queue_size: bool = False

    # LeRobot 0.7.0 optional features. Consumed by rollout.py when implemented.
    interpolation_multiplier: int = 1  # 1 = exact policy actions; 2 = optional interpolation.
    return_to_initial_position: bool = False  # No automatic return on stop by default.

    @property
    def environment_dt(self) -> float:
        """Base policy/action step (seconds)."""
        return 1.0 / self.fps

    @property
    def control_dt(self) -> float:
        """Robot command step; doubled control Hz only when interpolation is enabled."""
        return self.environment_dt / self.interpolation_multiplier

    @property
    def aggregate_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return get_aggregate_function(self.aggregate_fn_name)

    @property
    def policy_adapter_name(self) -> str:
        """Canonical adapter name; unknown names are left for future registrations."""
        return _BACKEND_ALIASES.get(self.backend, self.backend)

    def __post_init__(self) -> None:
        # Check configuration errors, not individual action values.
        if not self.backend.strip():
            raise ValueError("backend cannot be empty")
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if self.fps <= 0 or self.actions_per_chunk <= 0:
            raise ValueError("fps and actions_per_chunk must be positive")
        if self.zmq_timeout_ms <= 0:
            raise ValueError("zmq_timeout_ms must be positive")
        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("chunk_size_threshold must be between 0 and 1")
        if self.guard_steps < 0:
            raise ValueError("guard_steps must be nonnegative")
        if self.interpolation_multiplier not in (1, 2):
            raise ValueError("interpolation_multiplier must be 1 or 2")
        if not self.client_device:
            raise ValueError("client_device cannot be empty")
        for key in (self.front_camera_key, self.left_wrist_camera_key, self.right_wrist_camera_key):
            if not key:
                raise ValueError("camera keys must not be empty")
        # Validate the configured function now; no per-action intervention.
        get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotClientConfig:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return scalar settings (robot config is intentionally excluded as in legacy)."""
        return {
            "backend": self.backend,
            "server_address": self.server_address,
            "zmq_timeout_ms": self.zmq_timeout_ms,
            "task": self.task,
            "client_device": self.client_device,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "chunk_size_threshold": self.chunk_size_threshold,
            "aggregate_fn_name": self.aggregate_fn_name,
            "guard_steps": self.guard_steps,
            "front_camera_key": self.front_camera_key,
            "left_wrist_camera_key": self.left_wrist_camera_key,
            "right_wrist_camera_key": self.right_wrist_camera_key,
            "image_crop_params": self.image_crop_params,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "interpolation_multiplier": self.interpolation_multiplier,
            "return_to_initial_position": self.return_to_initial_position,
        }
