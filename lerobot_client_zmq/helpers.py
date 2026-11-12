"""Small data/timing helpers kept from the LeRobot 0.6.0 ZMQ client.

Policy-server observation formatting and action decoding belong in each
policy_adapter; ZMQ networking belongs in zmq_client.py.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


Action = torch.Tensor
RawObservation = dict[str, Any]
Observation = dict[str, torch.Tensor]


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_timestep(self) -> int:
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self) -> Action:
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self) -> RawObservation:
        return self.observation


@dataclass
class FPSTracker:
    target_fps: float
    first_timestamp: float | None = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        """Report the average observation sampling rate, as in the old client."""
        self.total_obs_count += 1
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp
        duration = current_timestamp - self.first_timestamp
        average = (self.total_obs_count - 1) / duration if duration > 1e-6 else 0.0
        return {"avg_fps": average, "target_fps": self.target_fps}

    def reset(self) -> None:
        self.first_timestamp = None
        self.total_obs_count = 0


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """Preserve legacy LeRobot console/file logger initialization."""
    from lerobot.utils.utils import init_logging

    log_file = None
    if log_to_file:
        Path("logs").mkdir(exist_ok=True)
        log_file = Path("logs") / f"{name}_{int(time.time())}.log"
    init_logging(log_file=log_file, display_pid=False)
    return logging.getLogger(name)


def visualize_action_queue_size(action_queue_size: list[int]) -> None:
    """Optional end-of-run queue visualization; not part of robot control."""
    if not action_queue_size:
        return
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.set_title("Action Queue Size Over Time")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action Queue Size")
    ax.set_ylim(0, max(action_queue_size) * 1.1 if max(action_queue_size) else 1)
    ax.grid(True, alpha=0.3)
    ax.plot(range(len(action_queue_size)), action_queue_size)
    plt.savefig("action_queue_states.png", dpi=150, bbox_inches="tight")
    plt.show()
