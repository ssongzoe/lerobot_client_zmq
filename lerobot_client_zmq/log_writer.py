"""Non-blocking JSONL run logs with the LeRobot 0.6 ZMQ client field names.

The controller enqueues small log records; one background thread owns all files.
Images are not written to JSONL, as in the legacy client.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from typing import Any

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


class RunLogWriter:
    """Write observations, raw actions, and dispatched actions to three JSONL files.

    Only the log thread performs disk I/O. ``close()`` drains the queue; callers
    must stop the inference worker before closing the writer.
    """

    def __init__(self, directory: str | Path = ".", *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.directory = Path(directory)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paths = {
            "observation": self.directory / f"observations_{self.run_id}.jsonl",
            "raw_action": self.directory / f"raw_actions_{self.run_id}.jsonl",
            "final_action": self.directory / f"final_actions_{self.run_id}.jsonl",
        }
        self._queue: SimpleQueue[tuple[str, dict[str, Any]] | None] = SimpleQueue()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._closed = False
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._write_loop, name="zmq-run-logger", daemon=True)
            self._thread.start()

    def _write_loop(self) -> None:
        try:
            # Files are opened only once. No repeated open/close in the 30 Hz loop.
            with (
                self.paths["observation"].open("a", encoding="utf-8") as obs_file,
                self.paths["raw_action"].open("a", encoding="utf-8") as raw_file,
                self.paths["final_action"].open("a", encoding="utf-8") as final_file,
            ):
                streams = {
                    "observation": obs_file,
                    "raw_action": raw_file,
                    "final_action": final_file,
                }
                while (item := self._queue.get()) is not None:
                    kind, payload = item
                    streams[kind].write(json.dumps(payload, default=_json_default) + "\n")
        except Exception as exc:
            self._error = exc
            logger.exception("JSONL log writer stopped")

    def _push(self, kind: str, payload: dict[str, Any]) -> None:
        if self.enabled and not self._closed:
            self._queue.put((kind, payload))

    def observation(
        self,
        *,
        timestamp: float,
        timestep: int,
        latest_action: int,
        queue_size: int,
        must_go: bool,
        raw_observation: dict[str, Any],
    ) -> None:
        # Preserve the old JSONL observation schema and omit camera images.
        scalars = {
            key: value
            for key, value in raw_observation.items()
            if key != "task" and not (getattr(value, "ndim", 0) >= 2)
        }
        self._push("observation", {
            "wall_time": timestamp,
            "timestep": timestep,
            "latest_action": latest_action,
            "queue_size": queue_size,
            "must_go": must_go,
            "observation": scalars,
        })

    def raw_action(
        self,
        *,
        timestamp: float,
        request_time_ms: float,
        observation_timestamp: float,
        observation_timestep: int,
        action_response: dict[str, Any],
    ) -> None:
        self._push("raw_action", {
            "wall_time": timestamp,
            "request_time_ms": request_time_ms,
            "observation_timestamp": observation_timestamp,
            "observation_timestep": observation_timestep,
            # Preserve the legacy field: GR00T's split keys yield None here.
            "actions": action_response.get("actions"),
        })

    def final_action(
        self,
        *,
        timestamp: float,
        policy_timestamp: float,
        timestep: int,
        action: dict[str, float],
        interpolated: bool = False,
    ) -> None:
        record = {
            "wall_time": timestamp,
            "policy_timestamp": policy_timestamp,
            "timestep": timestep,
            "action": action,
        }
        if interpolated:
            record["interpolated"] = True
        self._push("final_action", record)

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("JSONL log writer failed") from self._error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        self.check()
