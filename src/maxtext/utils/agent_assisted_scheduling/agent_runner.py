"""Launching the scheduling agent from the pass (rank 0 only).

`AgentRunner` is the seam: the pass hands it a working directory holding the module
and profile and a deadline, and gets back whether a plan file appeared. The only
implementation here shells out to headless Claude Code (`claude -p`). The agent is
untrusted input: the pass re-stamps, re-validates and re-applies whatever it
writes, and never executes anything the agent produced.

Tool restriction (best effort, enforced by the CLI): only Read, Write and the
`cli` command of this package are available, and anything else is denied rather
than prompted (`--permission-mode dontAsk`).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

CLI_MODULE = "maxtext.utils.agent_assisted_scheduling.cli"
SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
DEFAULT_TIMEOUT_S = 180.0  # 3 minutes of wall clock per module
_TAIL = 4000


@dataclass(frozen=True)
class AgentRequest:
  """Everything the agent needs for one module; paths are absolute."""

  module: str
  workdir: str
  module_path: str  # serialized HloModuleProto written by the pass
  profile_path: str  # FDO profile as text written by the pass
  plan_path: str  # where the agent must write its (unstamped) plan
  timeout_s: float = DEFAULT_TIMEOUT_S
  skills_dir: str = SKILLS_DIR
  python: str = sys.executable
  summary: str = ""  # one-line facts about the module for the prompt


@dataclass
class AgentResult:
  """What happened. `plan_written` is the only thing the pass trusts, and then only to re-validate it."""

  plan_written: bool
  duration_s: float
  returncode: Optional[int] = None
  timed_out: bool = False
  error: Optional[str] = None
  stdout_tail: str = ""
  stderr_tail: str = ""
  command: list[str] = field(default_factory=list)


class AgentRunner(Protocol):
  """Anything that can try to produce `request.plan_path` within `request.timeout_s`."""

  def run(self, request: AgentRequest) -> AgentResult:
    """Run the agent; never raise for agent failures, report them in the result."""


def cli_prefix(python: str = sys.executable) -> str:
  """The command prefix the agent is allowed to run."""
  return f"{python} -m {CLI_MODULE}"


def build_prompt(req: AgentRequest) -> str:
  """The task prompt. Kept short: the skills hold the knowledge."""
  cli = cli_prefix(req.python)
  return f"""You are scheduling the async collectives of one scheduled XLA:GPU module, module {req.module}.
{req.summary}

Inputs (read only): module {req.module_path}, FDO profile {req.profile_path}.
First read {req.skills_dir}/agent-assisted-scheduling-workflow/SKILL.md and follow it;
read the other three skills in that directory as needed.

You may ONLY use the Read and Write tools and commands starting with: {cli} ...
  {cli} analyze {req.module_path} --profile {req.profile_path} --json
  {cli} check-plan {req.module_path} --profile {req.profile_path} --plan <your plan file>
Do not run anything else, do not modify the inputs, do not use the network.

Goal: reduce total exposed collective time with legal reorders. Propose the full new order of each computation you change as
  {{"computations": {{"<computation>": ["<instruction>", ...]}}}}
Hard rules: every instruction stays in its own computation (no cross-computation moves); keep all data and control dependencies; never put a start next to its done.
Verify the plan with check-plan until it is ACCEPTED and improves the static or stream-sim total.
Then write it to {req.plan_path}.
If no safe move helps, write nothing. You have {int(req.timeout_s)} seconds in total; write the best verified
plan before then. Finish with one line: DONE <number of moved collectives>."""


class ClaudeCliRunner:
  """Runs `claude -p` as a subprocess with a hard wall-clock deadline."""

  def __init__(
      self,
      claude_bin: str = "claude",
      model: Optional[str] = None,
      max_budget_usd: Optional[float] = None,
      extra_args: tuple[str, ...] = (),
  ):
    self.claude_bin = claude_bin
    self.model = model
    self.max_budget_usd = max_budget_usd
    self.extra_args = tuple(extra_args)

  def command(self, req: AgentRequest) -> list[str]:
    """The exact command line (exposed for tests and logs)."""
    allowed = ["Read", "Write", f"Bash({cli_prefix(req.python)} *)"]
    cmd = [
        self.claude_bin,
        "-p",
        build_prompt(req),
        "--output-format",
        "json",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--tools",
        "Read,Write,Bash",
        "--allowedTools",
        *allowed,
        "--add-dir",
        req.workdir,
        "--add-dir",
        req.skills_dir,
    ]
    if self.model:
      cmd += ["--model", self.model]
    if self.max_budget_usd:
      cmd += ["--max-budget-usd", str(self.max_budget_usd)]
    return cmd + list(self.extra_args)

  def run(self, request: AgentRequest) -> AgentResult:
    """Run the CLI with a hard deadline; the process group is killed on timeout."""
    cmd = self.command(request)
    start = time.monotonic()
    try:
      proc = subprocess.Popen(  # pylint: disable=consider-using-with
          cmd,
          cwd=request.workdir,
          stdin=subprocess.DEVNULL,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          start_new_session=True,  # own process group so a timeout can kill the whole tree
      )
    except OSError as exc:
      return AgentResult(False, time.monotonic() - start, error=f"cannot start {self.claude_bin}: {exc}", command=cmd)
    timed_out = False
    try:
      out, err = proc.communicate(timeout=request.timeout_s)
    except subprocess.TimeoutExpired:
      timed_out = True
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      out, err = proc.communicate()
    return AgentResult(
        plan_written=os.path.exists(request.plan_path),
        duration_s=time.monotonic() - start,
        returncode=proc.returncode,
        timed_out=timed_out,
        error="timeout" if timed_out else (None if proc.returncode == 0 else f"agent exited {proc.returncode}"),
        stdout_tail=(out or "")[-_TAIL:],
        stderr_tail=(err or "")[-_TAIL:],
        command=cmd,
    )
