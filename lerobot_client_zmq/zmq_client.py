"""Shared Pi0.5 / GR00T ZMQ request client (legacy wire protocol).

The policy adapters own the model-specific observation and action schemas.
This module only encodes requests and decodes responses; it never changes
predicted action values, action queues, or robot commands.

Use one ZMQClient per thread. In particular, create/use/close the inference
client inside the inference worker. A ping on the main thread may use the
standalone ping_server() helper, which creates and closes its own socket.
"""
import io
from typing import Any

import msgpack
import numpy as np
import zmq


class MsgSerializer:
    """The existing servers' MsgPack + NumPy .npy wire encoding."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(
            data,
            object_hook=MsgSerializer._decode,
            strict_map_key=False,
        )

    @staticmethod
    def _encode(obj: Any) -> dict[str, Any]:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"Non-serializable type: {type(obj)}")

    @staticmethod
    def _decode(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


def normalize_zmq_server_address(server_address: str) -> str:
    """Accept both host:port and tcp://host:port, as in the old clients."""
    if server_address.startswith("tcp://"):
        return server_address

    host, sep, port = server_address.rpartition(":")
    if not sep or not host or not port:
        raise ValueError(
            f"ZMQ server_address must be 'host:port' or 'tcp://host:port', got {server_address!r}"
        )
    return f"tcp://{host}:{port}"


class ZMQClient:
    """Synchronous REQ/REP transport, intended for an async inference worker.

    A request may block up to timeout_ms waiting to send/receive. After a REQ
    timeout the socket may no longer be usable; close this client and let the
    caller decide whether to stop or construct a fresh client.
    """

    def __init__(self, server_address: str, timeout_ms: int = 30_000) -> None:
        self.address = normalize_zmq_server_address(server_address)
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)
        self._closed = False

    def _call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data

        try:
            self.sock.send(MsgSerializer.to_bytes(request))
            raw = self.sock.recv()
        except zmq.Again as exc:
            raise TimeoutError(f"Server response timeout (endpoint={endpoint})") from exc
        except zmq.ZMQError as exc:
            raise RuntimeError(f"ZMQ error: {exc}") from exc

        response = MsgSerializer.from_bytes(raw)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        """Legacy behavior: return False instead of raising on ping failure."""
        try:
            return bool(self._call("ping"))
        except Exception:
            return False

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the server's action dict, without modifying any values.

        Both existing servers accept {observation, options}; GR00T often
        responds with [action_dict, info_dict], while Pi0.5 may return a dict.
        """
        response = self._call(
            "get_action",
            {"observation": observation, "options": options},
        )
        if isinstance(response, (list, tuple)) and len(response) >= 1:
            return dict(tuple(response)[0])
        if isinstance(response, dict):
            return response
        raise RuntimeError(f"Unexpected action response type: {type(response)}")

    def reset(self, options: dict[str, Any] | None = None) -> Any:
        """Optional server reset; never called implicitly by the client."""
        return self._call("reset", {"options": options})

    def get_modality_config(self) -> Any:
        """Optional GR00T endpoint, retained for compatibility."""
        return self._call("get_modality_config")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.sock.close()
        self.ctx.term()


def ping_server(server_address: str, timeout_ms: int = 30_000) -> bool:
    """Ping on the calling thread without sharing its REQ socket with a worker."""
    client = ZMQClient(server_address, timeout_ms)
    try:
        return client.ping()
    finally:
        client.close()
