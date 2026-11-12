# LeRobot ZMQ Client

Asynchronous ZMQ inference client for LeRobot 0.7.x.

This project supports remote policy inference with RB-Y1 robots using
Pi0.5, GR00T N1.6, and extensible policy adapters.

## Requirements

Tested development environment:

- Python 3.12
- LeRobot 0.7.1
- lerobot-robot-rby1 0.3.1
- PyTorch
- NumPy
- PyZMQ
- MsgPack

The RB-Y1 robot plugin must be installed separately.

## Installation

Use an existing LeRobot environment.

```bash
cd ~/project/lerobot070/lerobot_client_zmq

uv pip install -e . --no-deps
```

## Rollout

Example for Pi0.5:

```bash
python -m lerobot_client_zmq.rollout \
  --robot.type=rby1 \
  --robot.address=127.0.0.1:50051 \
  --robot.model=m \
  --robot.version=1.3 \
  --backend=pi05 \
  --server_address=192.168.0.3:5566 \
  --task="Sort the flowers by color."
```

GR00T N1.6:

```bash
--backend=groot_n16
```

The production rollout requires the robot's configured cameras
and grippers to be available.

## Control Configuration

Default settings:

```text
fps                         30
chunk_size_threshold        0.5
aggregate_fn_name           weighted_average
guard_steps                 5
interpolation_multiplier    1
return_to_initial_position  false
```

### Action Chunk Aggregation

The client uses timestep-based action chunk aggregation.

Default aggregation:

```python
action = 0.3 * old_action + 0.7 * new_action
```


## Logs

The client records three JSONL files:

```text
observations_YYYYMMDD_HHMMSS.jsonl
raw_actions_YYYYMMDD_HHMMSS.jsonl
final_actions_YYYYMMDD_HHMMSS.jsonl
```

