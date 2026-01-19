"""
Workspace utilities for isolating lab runs.

A workspace is a dedicated directory under `data/lab-runs/` that holds:
- copied topology YAML files (derived from templates in `labs/`)
- any artifacts generated during lab usage (runtimes)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import shutil
import uuid


@dataclass
class Workspace:
    """Represents a per-user workspace directory."""
    id: str
    path: Path


def create_workspace(runs_dir: Path) -> Workspace:
    """
    Create a new unique workspace directory.

    Args:
        runs_dir: Base directory under which all workspaces are stored.

    Returns:
        A new Workspace with unique ID and path.
    """
    ws_id = uuid.uuid4().hex
    ws_path = runs_dir / f"ws-{ws_id}"
    ws_path.mkdir(parents=True, exist_ok=False)
    return Workspace(ws_id, ws_path)


def ensure_workspace_exists(runs_dir: Path, ws_id: str) -> Workspace:
    """
    Resolve and validate a workspace directory.

    Args:
        runs_dir: Base workspaces directory.
        ws_id: Workspace ID to check.

    Raises:
        FileNotFoundError: If the workspace directory does not exist.

    Returns:
        Workspace with validated ID and path.
    """
    ws_path = runs_dir / f"ws-{ws_id}"
    if not ws_path.exists() or not ws_path.is_dir():
        raise FileNotFoundError(f"Workspace not found: {ws_id}")
    return Workspace(ws_id, ws_path)


def resolve_template_yaml(labs_dir: Path, lab_name: str) -> Path:
    """
    Resolve a lab template YAML file by name.

    Args:
        labs_dir: Directory containing template YAML files.
        lab_name: Template name (stem), e.g. "testlab" -> "testlab.yml".

    Raises:
        ValueError: If `lab_name` contains suspicious path characters.
        FileNotFoundError: If the template file does not exist.

    Returns:
        Path to the resolved template YAML file.
    """
    # Allow only simple names, no separators or "..".
    if "/" in lab_name or ".." in lab_name or "\\" in lab_name:
        raise ValueError("Invalid lab name")
    f = labs_dir / f"{lab_name}.yml"
    if not f.exists():
        raise FileNotFoundError(f"Lab not found: {lab_name}")
    return f

def workspace_yaml_path(ws: Workspace, lab_name: str) -> Path:
    return ws.path / f"{lab_name}.yml"

def ensure_topology_in_workspace(template_yaml: Path, ws: Workspace, clab_name: str) -> Path:
    dest = ws.path / template_yaml.name
    if not dest.exists():
        shutil.copy2(template_yaml, dest)
    _rewrite_topology_name(dest, clab_name)
    return dest

def _rewrite_topology_name(yaml_path: Path, clab_name: str) -> None:
    txt = yaml_path.read_text(encoding="utf-8")
    pattern = r"(?m)^\s*name\s*:\s*.*$"
    if re.search(pattern, txt):
        txt = re.sub(pattern, f"name: {clab_name}", txt, count=1)
    else:
        txt = f"name: {clab_name}\n\n{txt}"
    yaml_path.write_text(txt, encoding="utf-8")


def copy_topology_into_workspace(template_yaml: Path, ws: Workspace) -> Path:
    """
    Copy a template YAML file into a workspace.

    This ensures each workspace operates on its own file copy, keeping shared
    templates immutable.

    Args:
        template_yaml: Source template file.
        ws: Target workspace.
    """
    dest = ws.path / template_yaml.name
    shutil.copy2(template_yaml, dest)
    return dest


def make_clab_name(ws: Workspace, topology_stem: str) -> str:
    """
    Build a short Containerlab deployment name for a workspace+topology.

    Args:
        ws: Workspace object.
        topology_stem: YAML stem (e.g. "testlab").
    """
    return f"ws-{ws.id[:8]}-{topology_stem}"