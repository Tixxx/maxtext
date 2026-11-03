"""Run 2 without recompiling: load a precompiled executable instead of compiling one module.

Three environment variables control it:

  COLLECTIVE_AOT_MODE     dump | load | off (default: off, nothing is patched)
  COLLECTIVE_AOT_PATH     the directory to dump into, or the bundle directory to load
  COLLECTIVE_AOT_MODULE   the module concerned (default "jit_train_step")

`jax._src.compiler.backend_compile_and_load` is wrapped; every module other than COLLECTIVE_AOT_MODULE compiles as usual.

  COLLECTIVE_AOT_MODE=dump  COLLECTIVE_AOT_PATH=<dir>
      The module compiles normally; process 0 then writes `<dir>/<module>.aot_header`, the metadata header of its
      serialized executable (see aot_bundle.py for why it is needed). Use the same directory as the module dump of
      COLLECTIVE_HLO_MODE=dump; `cli aot-build` finds the header there.

  COLLECTIVE_AOT_MODE=load  COLLECTIVE_AOT_PATH=<bundle dir>
      The bundle (`cli aot-build`) is checked (manifest, file hashes, jaxlib version, the partition count of this
      compile, and the number of arguments and results of the program being compiled against the header's specs)
      and the module is deserialized from it instead of compiled. Any mismatch or failure is logged with its reason
      and the module compiles normally.

The patch never raises into the compiler and never touches the JAX backend before jax.distributed is initialized.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from maxtext.utils.agent_assisted_scheduling import aot_bundle

_logger = logging.getLogger(__name__)
_NAME = "aot_executable"
MODE_ENV = "COLLECTIVE_AOT_MODE"
PATH_ENV = "COLLECTIVE_AOT_PATH"
MODULE_ENV = "COLLECTIVE_AOT_MODULE"
MODES = ("off", "dump", "load")


def _signature(mlir_module) -> Optional[tuple[int, int]]:
  """(arguments, results) of the program's `main`, or None if it cannot be read."""
  try:
    from jax._src.interpreters import mlir  # pylint: disable=import-outside-toplevel

    for op in mlir_module.body.operations:
      if op.operation.name == "func.func" and str(op.attributes["sym_name"]).strip('"') == "main":
        function_type = mlir.ir.FunctionType(op.attributes["function_type"].value)
        return len(function_type.inputs), len(function_type.results)
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return None


def _load(backend, mlir_module, executable_devices, options, name: str, directory: str):
  """The deserialized executable of the bundle in `directory`, or None (with the reason logged)."""
  try:
    manifest = aot_bundle.check(directory, name)
    partitions = options.executable_build_options.num_partitions
    if partitions != manifest.num_partitions:
      raise aot_bundle.BundleError(f"built for {manifest.num_partitions} partitions, compiling for {partitions}")
    with open(aot_bundle.header_path(directory, name), "rb") as f:
      header = f.read()
    expected = aot_bundle.header_spec_counts(header)
    signature = _signature(mlir_module)
    if signature is not None and signature != expected:
      raise aot_bundle.BundleError(
          f"the program has {signature[0]} arguments and {signature[1]} results, the bundle was made for {expected[0]} and {expected[1]}"
      )
    with open(aot_bundle.executable_path(directory, name), "rb") as f:
      payload = f.read()
    loaded = backend.deserialize_executable(header + payload, executable_devices, options)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    _logger.warning("%s [%s]: not using the bundle in %s (%s); compiling normally.", _NAME, name, directory, exc)
    return None
  _logger.info("%s [%s]: loaded the precompiled executable from %s (%d bytes); not compiling.", _NAME, name, directory, len(payload))
  return loaded


def _dump_header(backend, loaded, name: str, directory: str) -> None:
  try:
    header = aot_bundle.header_of(backend.serialize_executable(loaded))
    if not header:
      raise ValueError("the serialized executable has no metadata header")
    os.makedirs(directory, exist_ok=True)
    path = aot_bundle.header_path(directory, name)
    with open(path + ".tmp", "wb") as f:
      f.write(header)
    os.replace(path + ".tmp", path)
    _logger.info("%s [%s]: wrote the metadata header (%d bytes) to %s.", _NAME, name, len(header), path)
  except Exception:  # pylint: disable=broad-exception-caught
    _logger.exception("%s [%s]: could not write the metadata header.", _NAME, name)


def settings(env=None) -> tuple[str, str, str]:
  """(mode, path, module) from the environment; the mode is 'off' unless valid and a path is given."""
  env = os.environ if env is None else env
  mode = (env.get(MODE_ENV) or "off").strip().lower()
  path = env.get(PATH_ENV, "").strip()
  if mode not in MODES:
    _logger.warning("%s: %s=%r is not one of %s; off.", _NAME, MODE_ENV, mode, "/".join(MODES))
    mode = "off"
  if mode != "off" and not path:
    _logger.warning("%s: %s=%s needs %s; off.", _NAME, MODE_ENV, mode, PATH_ENV)
    mode = "off"
  return mode, path, env.get(MODULE_ENV, "jit_train_step")


def register() -> bool:
  """Wraps the compiler entry point; returns whether anything was patched."""
  mode, path, module = settings()
  if mode == "off":
    return False

  from jax._src import compiler  # pylint: disable=import-outside-toplevel

  original = compiler.backend_compile_and_load

  def patched(backend, mlir_module, executable_devices, options, host_callbacks):
    name = str(mlir_module.operation.attributes["sym_name"]).strip('"')
    if name != module:
      return original(backend, mlir_module, executable_devices, options, host_callbacks)
    if mode == "load":
      loaded = _load(backend, mlir_module, executable_devices, options, name, path)
      if loaded is not None:
        return loaded
    loaded = original(backend, mlir_module, executable_devices, options, host_callbacks)
    if mode == "dump" and backend.process_index() == 0:
      _dump_header(backend, loaded, name, path)
    return loaded

  compiler.backend_compile_and_load = patched
  _logger.info("%s: registered (mode=%s, path=%s, module=%s).", _NAME, mode, path, module)
  return True
