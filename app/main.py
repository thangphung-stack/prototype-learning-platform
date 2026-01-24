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
from fastapi import FastAPI, HTTPException, Response, Depends, Cookie
from pathlib import Path
import shutil
from pydantic import BaseModel
from typing import Optional
from app import clab, workspace
from app.student_auth import (
    StudentSessionStore,
    JoinCodeStore,
    SESSION_COOKIE_NAME
)

app = FastAPI()

join_codes = JoinCodeStore()
student_sessions = StudentSessionStore()

# Repository root (.../app/main.py) -> parent[1] points to project base folder).
BASE_DIR = Path(__file__).resolve().parents[1]
LABS_DIR = BASE_DIR / "labs"
RUNS_DIR = BASE_DIR / "data" / "lab-runs"

######################## ---- Student me endpoints ----############################
def get_me_workspace(session_id: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME) 
    ) -> workspace.Workspace:
    """
    Dependency: Resolve the student's workspace from the session cookie.
    the browser sends the cookie, we look up ws_id, and FastAPI injects `ws`.
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="not joined (missing session cookie)")
    ws_id = student_sessions.get_ws_id(session_id)
    if not ws_id:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    try:
        return workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
    except FileNotFoundError:
        raise HTTPException(status_code=401, detail="Workspace no longer exists")

@app.post("/debug/create-join-code")
def debug_create_join_code(count: int = 2):
    """
    Small helper to test first: pre-create N workspaces and return a join code 
    (In the real flow, a lecturer endpoint would do this.)
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    ws_ids = []
    for _ in range(count):
        ws = workspace.create_workspace(RUNS_DIR)
        ws_ids.append(ws.id)
    
    code = join_codes.create_code(ws_ids)
    return {"join_code": code, "workspace_ids": ws_ids}

class JoinRequest(BaseModel):
    code: str

@app.post("/join")
def join(req: JoinRequest, response: Response):
    """
    Exchange join code -> workspace assignment -> session cookie.

    After this, the frontend can just use /me/* endpoints without sending ws_id.
    """
    try:
        ws_id = join_codes.assign_workspace(req.code)
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid join code")
    except RuntimeError:
        raise HTTPException(status_code=409, detail="No workspaces left for code")

    session_id = student_sessions.create_session(ws_id, ttl_minutes=240)

    # Cookie = "you're logged in for this workspace" (simple prototype-style session)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True, 
        samesite="lax", 
        secure=False, 
        max_age=240 * 60 
    )
    return {"ok": True, "workspace_id": ws_id}

@app.get("/me/labs")
def me_labs(ws: workspace.Workspace = Depends(get_me_workspace)):
    """List templates + show whether each YAML already exists in the student's workspace."""
    template_labs = []
    for f in LABS_DIR.glob("*.yml"):
        in_ws = (ws.path / f.name).exists()
        template_labs.append({
            "id": f.stem, "filename":f.name, "in_workspace": in_ws
        })
    return {"workspace_id": ws.id, "labs": template_labs}

@app.post("/me/deploy/{lab_name}")
def me_deploy(lab_name: str, ws: workspace.Workspace = Depends(get_me_workspace)):
    """Deploy a lab for the current student (workspace comes from cookie)."""
    tpl = workspace.resolve_template_yaml(LABS_DIR, lab_name)
    clab_name = workspace.make_clab_name(ws, tpl.stem)
    ws_yaml = workspace.ensure_topology_in_workspace(tpl, ws, clab_name=clab_name)
    r = clab.deploy(ws_yaml, reconfigure=True)
    return {"workspace_id": ws.id, "clab_name": clab_name, **r.__dict__}

@app.post("/me/destroy/{lab_name}")
def me_destroy(lab_name: str, ws: workspace.Workspace = Depends(get_me_workspace)):
    """Destroy the student's lab deployment (if the workspace YAML exists)."""
    ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
    if not ws_yaml.exists():
        raise HTTPException(status_code=404, detail=f"Workspace topology not found: {ws_yaml}")
    r = clab.destroy(ws_yaml, cleanup=True)
    return {"workspace_id": ws.id, **r.__dict__}

@app.post("/me/reset/{lab_name}")
def me_reset(lab_name: str, ws: workspace.Workspace = Depends(get_me_workspace)):
    """Quick reset = (re)deploy with --reconfigure (nice for 'start fresh' button)."""
    tpl = workspace.resolve_template_yaml(LABS_DIR, lab_name)
    clab_name = workspace.make_clab_name(ws, tpl.stem)
    ws_yaml = workspace.ensure_topology_in_workspace(tpl, ws, clab_name=clab_name)
    r = clab.deploy(ws_yaml, reconfigure=True)
    return {"workspace_id": ws.id, "clab_name": clab_name, **r.__dict__}

@app.get("/me/status/{lab_name}")
def me_status(lab_name: str, ws: workspace.Workspace = Depends(get_me_workspace)):
    """Lightweight status: do we have the YAML, and does clab inspect succeed?"""
    ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
    if not ws_yaml.exists():
        return {
            "workspace_id": ws.id,
            "lab_name": lab_name,
            "yaml_present": False,
            "running": False,
            "inspect": None
        }
    r = clab.inspect(ws_yaml)
    return {
        "workspace_id": ws.id,
        "lab_name": lab_name,
        "yaml_present": True,
        "running": (r.return_code == 0),
        "inspect": r.__dict__
    }


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

#############-----------------the part down here is not used anymore, switch to me style endpoints ------------------#############
@app.get("/workspaces/{ws_id}/labs")
def list_workspace_labs(ws_id: str):
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        
        template_labs = []
        for f in LABS_DIR.glob("*.yml"):
            in_ws = (ws.path / f.name).exists()
            template_labs.append({
            "id": f.stem, "filename": f.name, "in_workspace": in_ws
            })

        return {"workspace_id": ws.id, "labs": template_labs}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

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

        clab_name = workspace.make_clab_name(ws, tpl.stem)
        ws_yaml = workspace.ensure_topology_in_workspace(tpl, ws, clab_name=clab_name)
        r = clab.deploy(ws_yaml, reconfigure=True)

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
        ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
        if not ws_yaml.exists():
            raise FileNotFoundError(f"Workspace topology not found: {ws_yaml}")

        r = clab.destroy(ws_yaml, cleanup=True)

        return {"workspace_id": ws.id, **r.__dict__}
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
        ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
        if not ws_yaml.exists():
            raise FileNotFoundError(f"Workspace topology not found: {ws_yaml}")

        r = clab.inspect(ws_yaml)
        return {"workspace_id": ws.id, **r.__dict__}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/workspaces/{ws_id}/status/{lab_name}")
def status(ws_id: str, lab_name: str):
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
        
        yaml_present = ws_yaml.exists()

        if not yaml_present:
            return {
                "workspace_id": ws.id,
                "lab_name": lab_name,
                "yaml_present": False,
                "running": False,
                "inspect": None
            }
        
        r = clab.inspect(ws_yaml)
        running = (r.return_code == 0)
        return {
            "workspace_id": ws.id,
            "lab_name": lab_name,
            "yaml_present": True,
            "running": running,
            "inspect": {
                "return_code": r.return_code,
                "stdout": r.stdout,
                "stderr": r.stderr
            }
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/workspaces/{ws_id}/reset/{lab_name}")
def reset(ws_id: str, lab_name: str):
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        tpl = workspace.resolve_template_yaml(LABS_DIR, lab_name)
        clab_name = workspace.make_clab_name(ws, tpl.stem)
        ws_yaml = workspace.ensure_topology_in_workspace(tpl, ws, clab_name=clab_name)

        r = clab.deploy(ws_yaml, reconfigure=True)
        return {
            "workspace_id": ws.id,
            "clab_name": clab_name,
            **r.__dict__
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/workspaces/{ws_id}")
def cleanup_workspace(ws_id: str):
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        shutil.rmtree(ws.path)
        return {"message": f"Workspace {ws_id} deleted successfully."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))