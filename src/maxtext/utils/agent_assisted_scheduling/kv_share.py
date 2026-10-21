"""Rank 0 decides, every rank applies the same bytes.

Every rank must compile a byte-identical executable, so exactly one rank (rank 0)
produces the plan and publishes it through the JAX distributed coordination
service key-value store; the others block on the key. An empty payload (after the
magic prefix) means "no plan", which is also what any failure publishes, so the
other ranks never wait for a rank that has already given up. A rank that cannot
read the key within the timeout raises `ShareTimeout` and the caller keeps XLA's
schedule.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

_MAGIC = b"AAS1"


class ShareTimeout(RuntimeError):
  """The plan did not arrive in time."""


class KvClient(Protocol):
  """The two coordination-service calls used (jax distributed client)."""

  def key_value_set_bytes(self, key: str, value: bytes) -> None:
    """Publish `value` under `key` (a key can be set once)."""

  def blocking_key_value_get_bytes(self, key: str, timeout_ms: int) -> bytes:
    """Wait for `key` and return its value."""


def share(
    client: Optional[KvClient],
    key: str,
    *,
    rank: int,
    produce: Callable[[], bytes],
    timeout_ms: int,
) -> bytes:
  """Plan bytes (b"" for none) as produced by rank 0, identical on every rank.

  With no client (single process) the caller is its own rank 0. `produce` runs on
  rank 0 only and must not raise; if it does, "no plan" is published.
  """
  if rank != 0:
    if client is None:
      raise ShareTimeout("no coordination client on a non-zero rank")
    try:
      raw = client.blocking_key_value_get_bytes(key, timeout_ms)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise ShareTimeout(f"no plan for {key} within {timeout_ms} ms: {exc}") from exc
    if not raw.startswith(_MAGIC):
      raise ShareTimeout(f"malformed payload for {key}")
    return raw[len(_MAGIC) :]
  try:
    payload = produce()
  except Exception:  # pylint: disable=broad-exception-caught
    payload = b""
  if client is not None:
    client.key_value_set_bytes(key, _MAGIC + payload)
  return payload
