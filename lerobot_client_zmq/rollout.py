"""LeRobot 0.7.0 RB-Y1 ZMQ rollout, preserving the 0.6.0 async control loop.

Run: python -m lerobot_client_zmq.rollout --robot.type=rby1 \\
         --backend=pi05 --server_address=127.0.0.1:5556 --task="..."

One control thread owns the robot. One worker thread owns its ZMQ REQ socket.
The legacy action-chunk aggregation is unchanged; action values are not clipped
or smoothed. Hold and server reset are intentionally *not* automatic.
"""

import logging
import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from typing import Any

import torch

# Register camera config choices for --robot.cameras; LeRobot 0.7 public API.
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import make_robot_from_config
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.import_utils import register_third_party_plugins

from .config import RobotClientConfig
from .helpers import TimedAction, TimedObservation, visualize_action_queue_size
from .log_writer import RunLogWriter
from .policy_adapter.base import ACTION_NAMES, BasePolicyAdapter
from .policy_adapter.groot_n16_adapter import GrootN16Adapter
from .policy_adapter.pi05_adapter import Pi05Adapter
from .zmq_client import ZMQClient, ping_server

logger = logging.getLogger(__name__)

# Adding a new model requires an adapter and one entry here, not a new loop.
POLICY_ADAPTERS: dict[str, type[BasePolicyAdapter]] = {
    "pi05": Pi05Adapter,
    "groot_n16": GrootN16Adapter,
}


class ZMQRollout:
    """Legacy async queue/dispatch with LeRobot 0.7 interpolation/timing opt-in."""

    def __init__(self, cfg: RobotClientConfig) -> None:
        self.cfg = cfg
        adapter_cls = POLICY_ADAPTERS.get(cfg.policy_adapter_name)
        if adapter_cls is None:
            raise ValueError(f"Unknown backend {cfg.backend!r}; available: {list(POLICY_ADAPTERS)}")
        self.adapter = adapter_cls(
            front_camera_key=cfg.front_camera_key,
            left_wrist_camera_key=cfg.left_wrist_camera_key,
            right_wrist_camera_key=cfg.right_wrist_camera_key,
        )

        self.robot = None
        self.writer: RunLogWriter | None = None
        self.shutdown_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.worker_error: BaseException | None = None
        self._robot_command_failed = False
        self._initial_position: dict[str, float] | None = None

        self.action_queue: deque[TimedAction] = deque()
        self.action_lock = threading.Lock()
        self.remote_observation_queue: Queue[tuple[TimedObservation, dict[str, Any]]] = Queue(maxsize=1)
        self.latest_action = -1
        self.action_chunk_size = -1
        self.must_go = threading.Event()
        self.must_go.set()
        self.queue_size_history: list[int] = []
        self.interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
        self._active_action: TimedAction | None = None

    def dispatch_allowed(self) -> bool:
        """Minimal 0.7-style dispatch gate: no command once stopped or faulted."""
        return not self.shutdown_event.is_set() and self.worker_error is None

    def _action_queue_size(self) -> int:
        with self.action_lock:
            return len(self.action_queue)

    def _observation_due(self) -> bool:
        # Legacy behavior: when the queue has fewer than the configured fraction
        # of the largest received chunk, continuously offer latest observations.
        with self.action_lock:
            return self.action_chunk_size <= 0 or (
                len(self.action_queue) / self.action_chunk_size <= self.cfg.chunk_size_threshold
            )

    def _merge_actions(self, incoming_actions: list[TimedAction]) -> None:
        """The legacy guard-steps and same-timestep aggregation, under one lock.

        As in the original implementation, steps <= latest_action + guard_steps
        are excluded from a *new* chunk even on the initial request.
        """
        with self.action_lock:
            existing = {a.timestep: a.action for a in self.action_queue}
            guard_until = self.latest_action + self.cfg.guard_steps
            kept = [
                a for a in self.action_queue
                if self.latest_action < a.timestep <= guard_until
            ]
            merged: list[TimedAction] = kept
            for action in incoming_actions:
                if action.timestep <= guard_until:
                    continue
                if action.timestep in existing:
                    action = TimedAction(
                        timestamp=action.timestamp,
                        timestep=action.timestep,
                        action=self.cfg.aggregate_fn(existing[action.timestep], action.action),
                    )
                merged.append(action)
            self.action_queue = deque(merged)

    def _publish_latest_observation(
        self, observation: TimedObservation, payload: dict[str, Any]
    ) -> None:
        try:
            self.remote_observation_queue.put_nowait((observation, payload))
        except Full:
            try:
                self.remote_observation_queue.get_nowait()
            except Empty:
                pass
            self.remote_observation_queue.put_nowait((observation, payload))

    def _capture_observation(self) -> None:
        raw = self.robot.get_observation()
        raw["task"] = self.cfg.task
        with self.action_lock:
            latest = self.latest_action
            size = len(self.action_queue)
            must_go = self.must_go.is_set() and size == 0
        observation = TimedObservation(
            timestamp=time.time(),
            timestep=max(latest, 0),
            observation=raw,
            must_go=must_go,
        )
        assert self.writer is not None
        self.writer.observation(
            timestamp=observation.timestamp,
            timestep=observation.timestep,
            latest_action=latest,
            queue_size=size,
            must_go=must_go,
            raw_observation=raw,
        )
        payload = self.adapter.build_observation(raw)
        self._publish_latest_observation(observation, payload)
        if must_go:
            self.must_go.clear()

    def _inference_worker(self) -> None:
        # ZMQ REQ sockets are NOT shared between threads.
        client = None
        try:
            client = ZMQClient(self.cfg.server_address, self.cfg.zmq_timeout_ms)
            while not self.shutdown_event.is_set():
                try:
                    observation, payload = self.remote_observation_queue.get(timeout=0.1)
                except Empty:
                    continue
                started = time.perf_counter()
                response = client.get_action(payload)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if self.shutdown_event.is_set():
                    break
                assert self.writer is not None
                self.writer.raw_action(
                    timestamp=time.time(),
                    request_time_ms=elapsed_ms,
                    observation_timestamp=observation.timestamp,
                    observation_timestep=observation.timestep,
                    action_response=response,
                )
                actions = self.adapter.convert_actions(
                    response,
                    timestamp=observation.timestamp,
                    timestep=observation.timestep,
                    environment_dt=self.cfg.environment_dt,
                    client_device=self.cfg.client_device,
                )
                if actions:
                    self._merge_actions(actions)
                    self.action_chunk_size = max(self.action_chunk_size, len(actions))
                    self.must_go.set()
                    logger.debug(
                        "ZMQ: received %d actions for observation #%d (%.1fms)",
                        len(actions), observation.timestep, elapsed_ms,
                    )
        except Exception as exc:
            self.worker_error = exc
            self.shutdown_event.set()
            logger.exception("ZMQ inference worker failed")
        finally:
            if client is not None:
                client.close()

    def _dispatch(self, timer: CycleTimer) -> bool:
        """Pop one policy action per policy step; send only when permitted."""
        if not self.dispatch_allowed():
            self.interpolator.reset()
            return False

        if self.interpolator.needs_new_action():
            with self.action_lock:
                self.queue_size_history.append(len(self.action_queue))
                self._active_action = self.action_queue.popleft() if self.action_queue else None
            if self._active_action is None:
                timer.note_starved_tick()
                return False
            self.interpolator.add(self._active_action.action.detach().cpu())

        predicted = self.interpolator.get()
        if predicted is None or self._active_action is None:
            timer.note_starved_tick()
            return False
        if not self.dispatch_allowed():
            self.interpolator.reset()
            return False

        # No clipping or intervention: mapping keys is the adapter's only change.
        command = self.adapter.to_robot_action(predicted)
        if not self.dispatch_allowed():
            self.interpolator.reset()
            return False
        try:
            self.robot.send_action(command)
        except Exception:
            self._robot_command_failed = True
            raise

        assert self.writer is not None
        self.writer.final_action(
            timestamp=time.time(),
            policy_timestamp=self._active_action.timestamp,
            timestep=self._active_action.timestep,
            action=command,
            interpolated=not self.interpolator.emitted_policy_action,
        )
        if self.interpolator.emitted_policy_action:
            with self.action_lock:
                self.latest_action = self._active_action.timestep
        return True

    def _return_to_initial(self) -> None:
        """Opt-in return, using the 0.7.0 rollout's 3s/50Hz linear position ramp.

        This is intentionally not executed after a failed robot command.
        """
        if self._initial_position is None or self._robot_command_failed:
            return
        from lerobot.utils.robot_utils import precise_sleep

        current = self.adapter.normalize_observation(self.robot.get_observation())
        start = {
            self.adapter._action_keys[name]: float(current[name])
            for name in ACTION_NAMES
        }
        steps = 150  # 3 seconds at 50 Hz, as in LeRobot 0.7's rollout helper
        logger.info("Returning to captured initial position (3s, 50Hz)")
        for step in range(1, steps + 1):
            fraction = step / steps
            command = {
                key: start[key] * (1 - fraction) + target * fraction
                for key, target in self._initial_position.items()
            }
            self.robot.send_action(command)
            precise_sleep(1.0 / 50.0)

    def run(self) -> None:
        from lerobot.robots.utils import check_control_rate

        try:
            # Same order as 0.6: construct robot -> connect -> validate -> ping.
            self.robot = make_robot_from_config(self.cfg.robot)
            self.robot.connect()
            self.adapter.validate_robot(self.robot)
            check_control_rate(self.robot, self.cfg.fps, multiplier=self.cfg.interpolation_multiplier)
            if self.cfg.return_to_initial_position:
                observation = self.robot.get_observation()
                state = self.adapter.normalize_observation(observation)
                self._initial_position = {
                    self.adapter._action_keys[name]: float(state[name])
                    for name in ACTION_NAMES
                }
            logger.info("Robot connected and ready")

            if not ping_server(self.cfg.server_address, self.cfg.zmq_timeout_ms):
                raise ConnectionError(f"ZMQ ping failed: {self.cfg.server_address}")
            logger.info("ZMQ ping OK: %s", self.cfg.server_address)

            self.writer = RunLogWriter(enabled=False)
            self.worker = threading.Thread(
                target=self._inference_worker, name="zmq-inference", daemon=True
            )
            self.worker.start()

            timer = CycleTimer(
                self.cfg.fps,
                self.cfg.interpolation_multiplier,
                records_data=False,
            )
            logger.info("Control loop started: %g policy Hz, %g command Hz", self.cfg.fps,
                        self.cfg.fps * self.cfg.interpolation_multiplier)
            try:
                while self.dispatch_allowed():
                    new_cycle = self.interpolator.needs_new_action()
                    timer.tick(new_cycle=new_cycle)
                    # Preserve legacy order: action first, observation second.
                    with timer.section("action"):
                        self._dispatch(timer)
                    if self.dispatch_allowed() and new_cycle and self._observation_due():
                        with timer.section("observation"):
                            self._capture_observation()
                    if self.writer is not None:
                        self.writer.check()
                    timer.wait()
            finally:
                timer.log_run_summary()
            if self.worker_error is not None:
                raise RuntimeError("ZMQ inference worker failed") from self.worker_error
        finally:
            self.stop()

    def stop(self) -> None:
        # Revoke dispatch before any disconnect/return movement.
        self.shutdown_event.set()
        if self.worker is not None:
            # An in-flight ZMQ call can take up to zmq_timeout_ms; worker is daemon.
            self.worker.join(timeout=1.0)
        try:
            if self.robot is not None and self.robot.is_connected:
                try:
                    if self.cfg.return_to_initial_position and not self._robot_command_failed:
                        self._return_to_initial()
                finally:
                    self.robot.disconnect()
                    logger.info("Robot disconnected")
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.cfg.debug_visualize_queue_size:
                visualize_action_queue_size(self.queue_size_history)


@parser.wrap()
def rollout(cfg: RobotClientConfig) -> None:
    # LeRobot 0.7 parser supports --robot.type and third-party hardware plugins.
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    try:
        ZMQRollout(cfg).run()
    except KeyboardInterrupt:
        logger.info("Stopped by user")


def main() -> None:
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
