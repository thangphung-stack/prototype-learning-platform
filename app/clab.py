"""
Small wrapper around the `containerlab` CLI.

This module just calls the `clab` binary for deploy/destroy/inspect and captures
stdout/stderr so the API can return a structured result instead of dealing with
raw subprocess output.

Return a `ClabResult` so callers always get the same shape back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional, Sequence
import json


@dataclass
class ClabResult:
    """Result of a `clab` subprocess execution"""
    return_code: int
    stdout: str
    stderr: str


def _run(args: Sequence[str], cwd: Optional[Path] = None) -> ClabResult:
    """
    Execute a subprocess command and capture its output.

    Args:
        args: Command and arguments, e.g. ["clab", "inspect", "-t", "..."].
        cwd: Optional working directory for the process.

    Returns:
        ClabResult containing return code, stdout and stderr.
    """
    p = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        capture_output=True, #capture both stdout and stderr
        text=True, #decode bytes to str using system encoding
    )
    return ClabResult(p.returncode, p.stdout, p.stderr)


def deploy(topology_file: Path, reconfigure: bool = False) -> ClabResult:
    """
    Deploy a Containerlab topology.

    Args:
        topology_file: Path to a topology YAML file.
        no more name.
        reconfigure: If True, pass `--reconfigure` to clab.

    Returns:
        ClabResult with the subprocess result.
    """
    if not topology_file.exists():
        return ClabResult(1, "", f"Topology file not found: {topology_file}")

    cmd = ["clab","-t", str(topology_file),"deploy"]                    
    if reconfigure:
        cmd += ["--reconfigure"] 
    #run in the topology dir so relative paths work
    return _run(cmd, cwd=topology_file.parent)


def destroy(topology_file: Path, name: Optional[str] = None, cleanup: bool = False) -> ClabResult:
    """
    Destroy a deployed Containerlab topology.

    Args:
        topology_file: Path to the topology YAML file used for deployment.
        name: Optional explicit lab name (Containerlab `--name`).
        cleanup: if true, remove the artifacts (runtime)

    Returns:
        ClabResult 
    """    
    if not topology_file.exists():
        return ClabResult(1, "", f"Topology file not found: {topology_file}")

    cmd = ["clab", "-t", str(topology_file), "destroy"]                  
    if cleanup:
        cmd += ["--cleanup"]

    return _run(cmd, cwd=topology_file.parent)


def inspect(topology_file: Path, name: Optional[str] = None) -> ClabResult:
    """
    Inspect a Containerlab topology (runtime state).

    Args: topology_file, name

    Returns:
        ClabResult 
    """    
    if not topology_file.exists():
        return ClabResult(1, "", f"Topology file not found: {topology_file}")

    cmd = ["clab", "-t", str(topology_file), "inspect"]

    return _run(cmd, cwd=topology_file.parent)

def inspect_json(topology_file: Path) -> dict:
    if not topology_file.exists():
        return {"error": f"Topology file not found: {topology_file}"}

    cmd = ["clab", "-t", str(topology_file), "inspect", "--format", "json"]
    r = _run(cmd, cwd=topology_file.parent)

    if r.return_code != 0:
        return {"error": r.stderr or "inspect failed", "raw": r.__dict__}

    try:
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": f"Failed to parse inspect json: {e}", "raw_stdout": r.stdout}