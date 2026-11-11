"""The files that let a later run load an optimized, scheduled module instead of compiling it.

A bundle is a directory:

    <module>.aot_header       metadata header of a normal compile of the module (written by aot_executable.py)
    <module>.aot_executable   what `xla_compile` makes of the optimized, scheduled module
    manifest.json             what the two files are, so a stale or damaged bundle is not loaded

`build` makes it from a COLLECTIVE_AOT_MODE=dump directory and the optimized module (`cli aot-build`); `check`
is what the loader runs before trusting it (`cli aot-check`). Both are jax-free apart from the jaxlib version check.

Why a header at all: `xla_compile` writes the PjRt executable only, and JAX's loader wants it behind a
size-delimited SerializedXlaExecutableMetadata message that holds the runtime ABI and a spec (shape, dtype, layout,
sharding, memory kind, donation) per parameter and output. The optimized module has the same interface as the one it
was made from, so the header of any normal compile of the program fits it.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
from typing import Optional

from maxtext.utils.agent_assisted_scheduling.profile import iter_fields

SCHEMA = 1
MANIFEST = "manifest.json"
HEADER_SUFFIX = ".aot_header"
EXECUTABLE_SUFFIX = ".aot_executable"
XLA_COMPILE_ENV = "XLA_COMPILE_BIN"
DEVICES_PER_HOST_ENV = "COLLECTIVE_AOT_DEVICES_PER_HOST"


class BundleError(Exception):
  """A bundle cannot be built or must not be loaded; the message says why."""


@dataclasses.dataclass(frozen=True)
class Manifest:
  """What a bundle holds and was built from."""

  module: str
  header_sha256: str
  executable_sha256: str
  optimized_module_sha256: str
  num_parameters: int
  num_outputs: int
  num_partitions: int
  jaxlib_version: str
  source_dump: str

  def to_json(self) -> str:
    return json.dumps({"schema": SCHEMA, **dataclasses.asdict(self)}, indent=2, sort_keys=True) + "\n"


def header_path(directory: str, module: str) -> str:
  return os.path.join(directory, module + HEADER_SUFFIX)


def executable_path(directory: str, module: str) -> str:
  return os.path.join(directory, module + EXECUTABLE_SUFFIX)


def sha256_file(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def header_of(serialized: bytes) -> Optional[bytes]:
  """The leading size-delimited metadata message of an IFRT-serialized executable, with its length prefix."""
  length, shift, i = 0, 0, 0
  while i < len(serialized):
    byte = serialized[i]
    length |= (byte & 0x7F) << shift
    i += 1
    shift += 7
    if not byte & 0x80:
      return serialized[: i + length] if i + length <= len(serialized) else None
  return None


def header_spec_counts(header: bytes) -> tuple[int, int]:
  """(parameters, outputs) a header describes: its `parameter_specs` (field 6) and `output_specs` (field 5)."""
  length, shift, i = 0, 0, 0
  while i < len(header):
    byte = header[i]
    length |= (byte & 0x7F) << shift
    i += 1
    shift += 7
    if not byte & 0x80:
      break
  params = outputs = 0
  for number, _, _ in iter_fields(header[i : i + length]):
    params += number == 6
    outputs += number == 5
  return params, outputs


def _jaxlib_version() -> str:
  try:
    import jaxlib  # pylint: disable=import-outside-toplevel

    return str(jaxlib.__version__)
  except Exception:  # pylint: disable=broad-exception-caught
    return ""


def _strip_entry_layout_and_dump(config_text: str) -> str:
  """The module config without `entry_computation_layout` and `xla_dump_to`.

  The dumped config's entry layout is the global one, before SPMD partitioning; the scheduled module is per device, so
  XLA must take the layout from the module. `xla_dump_to` would make `xla_compile` write into the original run's dump.
  """
  lines = config_text.split("\n")
  out: list[str] = []
  skipping = False
  for line in lines:
    if skipping:
      skipping = line != "}"
      continue
    if line.startswith("entry_computation_layout {"):
      skipping = True
      continue
    if re.match(r"\s+xla_dump_to:", line):
      continue
    out.append(line)
  return "\n".join(out)


def _devices_per_host(devices_per_host: Optional[int]) -> int:
  """GPUs per host for the deviceless topology: the argument, else $COLLECTIVE_AOT_DEVICES_PER_HOST, else 0 (unset)."""
  if devices_per_host is None:
    raw = os.environ.get(DEVICES_PER_HOST_ENV, "").strip()
    if not raw:
      return 0
    try:
      devices_per_host = int(raw)
    except ValueError:
      devices_per_host = 0
    if devices_per_host <= 0:
      raise BundleError(f"{DEVICES_PER_HOST_ENV} must be a positive integer, got {raw!r}")
  return max(devices_per_host, 0)


def _dumped_compile_files(dump_dir: str, module: str, profile: Optional[str]) -> tuple[str, Optional[str], str]:
  """(module config, autotune results or None, GPU target config) of the compile the optimized module came from."""
  module_id = None
  if profile:
    m = re.match(r"module_(\d+)\.", os.path.basename(profile))
    module_id = m.group(1) if m else None
  if module_id is not None:
    config = os.path.join(dump_dir, f"module_{module_id}.{module}.config.pbtxt")
    configs = [config] if os.path.exists(config) else []
  else:
    configs = []
  if not configs:
    configs = sorted(glob.glob(os.path.join(dump_dir, f"module_*.{module}.config.pbtxt")), key=os.path.getmtime)[-1:]
  if not configs:
    raise BundleError(f"no module_*.{module}.config.pbtxt in {dump_dir}")
  autotune = configs[0].replace(".config.pbtxt", ".autotune_results.pbtxt")
  target = configs[0].replace(".config.pbtxt", ".gpu_target_config.pbtxt")
  if not os.path.exists(target):
    raise BundleError(f"no {os.path.basename(target)} in {dump_dir}: run 1 needs --xla_dump_to")
  return configs[0], autotune if os.path.exists(autotune) else None, target


_CUDNN_DEVICELESS_RE = re.compile(r"^(\s*)xla_gpu_cudnn_deviceless_compilation_mode:.*$", re.M)
_CUDNN_DEVICELESS_AUTO = "CUDNN_DEVICELESS_COMPILATION_AUTO"


def _enable_deviceless_cudnn(config_text: str) -> str:
  """The module config with cuDNN graphs compiled without a device.

  Run 1's config says `CUDNN_DEVICELESS_COMPILATION_DISABLED`, and then a compile with no device skips building the
  cuDNN graphs (fused attention) altogether: it succeeds, and the executable fails at its first step with "No
  serialized cuDNN graph with fingerprint ...". Needs cuDNN 9.23 or newer.
  """
  setting = f"xla_gpu_cudnn_deviceless_compilation_mode: {_CUDNN_DEVICELESS_AUTO}"
  if _CUDNN_DEVICELESS_RE.search(config_text):
    return _CUDNN_DEVICELESS_RE.sub(lambda m: f"{m.group(1)}{setting}", config_text)
  if "debug_options {" in config_text:
    return config_text.replace("debug_options {", f"debug_options {{\n  {setting}", 1)
  return config_text.rstrip("\n") + f"\ndebug_options {{\n  {setting}\n}}\n"


def _num_partitions(config_text: str) -> int:
  m = re.search(r"^num_partitions:\s*(\d+)", config_text, re.M)
  return int(m.group(1)) if m else 1


def build(
    dump_dir: str,
    module: str,
    optimized_module: str,
    out_dir: str,
    profile: Optional[str] = None,
    xla_compile: Optional[str] = None,
    devices_per_host: Optional[int] = None,
) -> Manifest:
  """Compiles `optimized_module` with `xla_compile` and writes the bundle for `module` into `out_dir`.

  `dump_dir` is the COLLECTIVE_AOT_MODE=dump directory of run 1 (it holds the header and the XLA dumps of the
  compile). `profile` is the FDO profile file `optimize-dump` used; it names the compile whose config to reuse.
  The compile needs no GPU: it uses run 1's GPU target config. `devices_per_host` (default
  $COLLECTIVE_AOT_DEVICES_PER_HOST) is the number of GPUs on each node and decides which collectives count as local to
  a host; unset, `xla_compile` assumes every partition is on one host, which is right for a single node only. It
  needs an `xla_compile` with the `--num_devices_per_host` flag.
  """
  header = header_path(dump_dir, module)
  if not os.path.exists(header):
    raise BundleError(f"no {os.path.basename(header)} in {dump_dir}: run 1 needs COLLECTIVE_AOT_MODE=dump")
  binary = xla_compile or os.environ.get(XLA_COMPILE_ENV) or shutil.which("xla_compile")
  if not binary or not os.path.exists(binary):
    raise BundleError(
        "xla_compile not found: pass --xla-compile or set XLA_COMPILE_BIN; build it with "
        "`build-jax.sh --no-install --extra-targets=@xla//xla/service:xla_compile --extra-target-dest=DIR`"
    )
  config, autotune, target_config = _dumped_compile_files(dump_dir, module, profile)
  with open(config, encoding="utf-8") as f:
    config_text = f.read()

  os.makedirs(out_dir, exist_ok=True)
  compile_config = os.path.join(out_dir, module + ".compile_config.pbtxt")
  with open(compile_config, "w", encoding="utf-8") as f:
    f.write(_enable_deviceless_cudnn(_strip_entry_layout_and_dump(config_text)))
  executable = executable_path(out_dir, module)
  cmd = [
      binary,
      f"--module_file={optimized_module}",
      f"--module_config_file={compile_config}",
      f"--gpu_target_config={target_config}",
      "--platform=gpu",
      f"--output_file={executable}",
  ]
  if autotune:
    cmd.append(f"--autotune_results={autotune}")
  per_host = _devices_per_host(devices_per_host)
  if per_host:
    cmd.append(f"--num_devices_per_host={per_host}")
  result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if result.returncode != 0 or not os.path.exists(executable):
    tail = "\n".join(result.stderr.strip().splitlines()[-5:])
    raise BundleError(f"xla_compile failed (exit {result.returncode}):\n{tail}")

  shutil.copyfile(header, header_path(out_dir, module))
  with open(header, "rb") as f:
    n_params, n_outputs = header_spec_counts(f.read())
  manifest = Manifest(
      module=module,
      header_sha256=sha256_file(header_path(out_dir, module)),
      executable_sha256=sha256_file(executable),
      optimized_module_sha256=sha256_file(optimized_module),
      num_parameters=n_params,
      num_outputs=n_outputs,
      num_partitions=_num_partitions(config_text),
      jaxlib_version=_jaxlib_version(),
      source_dump=os.path.abspath(dump_dir),
  )
  with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as f:
    f.write(manifest.to_json())
  return manifest


def check(directory: str, module: str, current_jaxlib: Optional[str] = None) -> Manifest:
  """The manifest of a bundle that is safe to load for `module`; raises BundleError saying why it is not."""
  path = os.path.join(directory, MANIFEST)
  try:
    with open(path, encoding="utf-8") as f:
      raw = json.load(f)
  except (OSError, ValueError) as exc:
    raise BundleError(f"cannot read {path}: {exc}") from exc
  if raw.get("schema") != SCHEMA:
    raise BundleError(f"{path}: schema {raw.get('schema')!r}, expected {SCHEMA}")
  try:
    manifest = Manifest(**{k: v for k, v in raw.items() if k != "schema"})
  except TypeError as exc:
    raise BundleError(f"{path}: {exc}") from exc
  if manifest.module != module:
    raise BundleError(f"the bundle is for module {manifest.module!r}, not {module!r}")
  for label, file, expected in (
      ("header", header_path(directory, module), manifest.header_sha256),
      ("executable", executable_path(directory, module), manifest.executable_sha256),
  ):
    if not os.path.exists(file):
      raise BundleError(f"the {label} {file} is missing")
    if sha256_file(file) != expected:
      raise BundleError(f"the {label} {file} does not match the manifest (changed or damaged)")
  jaxlib_version = _jaxlib_version() if current_jaxlib is None else current_jaxlib
  if manifest.jaxlib_version and jaxlib_version and manifest.jaxlib_version != jaxlib_version:
    raise BundleError(f"built with jaxlib {manifest.jaxlib_version}, running {jaxlib_version}")
  return manifest
