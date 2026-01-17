"""
FastAPI application for managing Containerlab-based labs.

This service exposes endpoints to:
- list available lab templates (YAML files)
- create per-user workspaces
- deploy/destroy/inspect labs inside a workspace

Directories structure:
- labs/ contains template YAML topologies
- data/lab-runs/ contains workspace folders (runtime) and per-workspace YAML copies
"""
from fastapi import FastAPI, HTTPException
from pathlib import Path

from app import clab, workspace

app = FastAPI()

# Repository root (.../app/main.py) -> parent[1] points to project base folder).
BASE_DIR = Path(__file__).resolve().parents[1]
LABS_DIR = BASE_DIR / "labs"
RUNS_DIR = BASE_DIR / "data" / "lab-runs"


@app.get("/")
def home():
    return {"message": "hello from prototype-learning-platform"}


@app.get("/list")
def list_labs():
    """Return all available lab templates found in the labs directory."""
    labs = [{"id": f.stem, "filename": f.name} for f in LABS_DIR.glob("*.yml")]
    return {"labs": labs}


@app.post("/workspaces")
def create_ws():
    """
    Create a new workspace directory.

    A workspace is an isolated folder that contains copies of selected lab templates,
    allowing users to deploy/inspect/destroy without modifying shared templates.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ws = workspace.create_workspace(RUNS_DIR)
    return {"workspace_id": ws.id}


@app.post("/workspaces/{ws_id}/deploy/{lab_name}")
def deploy(ws_id: str, lab_name: str):
    """
    Deploy a lab into the given workspace.

    Steps:
    - verify workspace exists
    - resolve the requested template YAML
    - copy template YAML into the workspace
    - deploy via containerlab using a deterministic lab name
    """
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        tpl = workspace.resolve_template_yaml(LABS_DIR, lab_name)
        ws_yaml = workspace.copy_topology_into_workspace(tpl, ws)

        clab_name = workspace.make_clab_name(ws, tpl.stem)
        r = clab.deploy(ws_yaml, name=clab_name)

        return {"workspace_id": ws.id, "clab_name": clab_name, **r.__dict__}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/workspaces/{ws_id}/destroy/{lab_name}")
def destroy(ws_id: str, lab_name: str):
    """
    Destroy a lab deployment within the workspace.

    This expects the workspace copy of the topology YAML to exist, ensuring
    users do not destroy deployments using incorrect file paths.
    """
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        ws_yaml = ws.path / f"{lab_name}.yml"
        if not ws_yaml.exists():
            raise FileNotFoundError(f"Workspace topology not found: {ws_yaml}")

        clab_name = workspace.make_clab_name(ws, lab_name)
        r = clab.destroy(ws_yaml, name=clab_name)

        return {"workspace_id": ws.id, "clab_name": clab_name, **r.__dict__}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/workspaces/{ws_id}/inspect/{lab_name}")
def inspect(ws_id: str, lab_name: str):
    """
    Inspect a deployed lab within the workspace.

    The response includes the raw stdout/stderr from `clab inspect`, which can be
    parsed by the client if needed.
    """
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        ws_yaml = ws.path / f"{lab_name}.yml"
        if not ws_yaml.exists():
            raise FileNotFoundError(f"Workspace topology not found: {ws_yaml}")

        clab_name = workspace.make_clab_name(ws, lab_name)
        r = clab.inspect(ws_yaml, name=clab_name)

        return {"workspace_id": ws.id, "clab_name": clab_name, **r.__dict__}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))