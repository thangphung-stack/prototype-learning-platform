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
from fastapi import FastAPI, HTTPException, Response, Depends, Cookie, Request
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
from fastapi.staticfiles import StaticFiles
import yaml
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import select
from datetime import datetime, timedelta, timezone
import secrets

from app.db import engine, SessionLocal
from app.models import Lecturer, Booking, WorkspaceSlot, StudentSession
from app import security

app = FastAPI()

join_codes = JoinCodeStore()
student_sessions = StudentSessionStore()

# Repository root (.../app/main.py) -> parent[1] points to project base folder).
BASE_DIR = Path(__file__).resolve().parents[1]
LABS_DIR = BASE_DIR / "labs"
RUNS_DIR = BASE_DIR / "data" / "lab-runs"
UI_DIR = BASE_DIR / "ui"

SESSION_COOKIE = "student_session"

app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup():
    from app.db import Base
    Base.metadata.create_all(bind=engine)

#######################------------------Teacher authentication------------------#######################
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@app.post("/auth/register")
def register(email: str, password: str, db: Session = Depends(get_db)):
    email_existing = db.execute(select(Lecturer).where(Lecturer.email == email)).scalar_one_or_none()
    if email_existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    
    lec = Lecturer(email=email, password_hash=security.hash_password(password))
    db.add(lec)
    db.commit()
    return {"ok": True}

@app.post("/auth/token")
def token(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    lec = db.execute(select(Lecturer).where(Lecturer.email == form.username)).scalar_one_or_none()
    if not lec or not security.verify_password(form.password, lec.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = security.create_access_token(subject=str(lec.id))
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_lecturer(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> Lecturer:
    try:
        payload = security.decode_token(token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    lec_id = int(payload.get("sub", "0"))
    lec = db.get(Lecturer, lec_id)
    if not lec:
        raise HTTPException(status_code=401, detail="Invalid token")
    return lec

def make_join_code() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz1234567890"
    return "".join(secrets.choice(alphabet) for _ in range(8))

@app.post("/teacher/bookings")
def create_booking(
    seats: int,
    db : Session = Depends(get_db),
    lec: Lecturer = Depends(get_current_lecturer)
):
    if seats <= 0 or seats > 200:
        raise HTTPException(status_code=400, detail="Invalid number of seats")
    
    join_code = make_join_code()
    booking = Booking(
        lecturer_id=lec.id,
        join_code=join_code,
        seats=seats,
        starts_at=None,
        ends_at=None
    )
    db.add(booking)
    db.flush()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(seats):
        ws = workspace.create_workspace(RUNS_DIR)
        db.add(WorkspaceSlot(booking_id=booking.id, ws_id=ws.id))
    
    db.commit()

    return {
        "booking_id": booking.id,
        "join_code": join_code,
        "join_url": "/ui/",
        "seats": seats,
    }

def get_student_session_id(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)

def now_utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

@app.post("/join") 
def join(code: str, response: Response, request: Request, db: Session = Depends(get_db)):
    #rejoin logic if cookies already exists. 
    sid = get_student_session_id(request)
    if sid:
        sess = db.execute(select(StudentSession).where(StudentSession.session_id == sid)).scalar_one_or_none()
        if sess and sess.expires_at > now_utc_naive():
            return {"ok": True}
    
    #validate join code (booking lookup)
    booking = db.execute(select(Booking).where(Booking.join_code == code)).scalar_one_or_none()
    if not booking:
        raise HTTPException(status_code=404, detail="Invalid join code")
    
    slot = db.execute(
        select(WorkspaceSlot)
        .where(WorkspaceSlot.booking_id == booking.id, WorkspaceSlot.assigned_at.is_(None))
        .limit(1)
    ).scalar_one_or_none()

    if not slot:
        raise HTTPException(status_code=409, detail="No workspaces left for this booking")
    
    sid = secrets.token_hex(16)
    slot.assigned_at = now_utc_naive()
    slot.assigned_session_id = sid

    sess = StudentSession(
        session_id=sid,
        workspace_slot_id=slot.id,
        created_at=now_utc_naive(),
        expires_at=now_utc_naive() + timedelta(hours=6),
    )
    db.add(sess)
    db.commit()

    response.set_cookie(
        key= SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=6 * 3600,
    )
    return {"ok": True}

######################## ---- Student me endpoints ----############################
# def get_me_workspace(session_id: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME) 
#     ) -> workspace.Workspace:
#     """
#     Dependency: Resolve the student's workspace from the session cookie.
#     the browser sends the cookie, we look up ws_id, and FastAPI injects `ws`.
#     """
#     if not session_id:
#         raise HTTPException(status_code=401, detail="not joined (missing session cookie)")
#     ws_id = student_sessions.get_ws_id(session_id)
#     if not ws_id:
#         raise HTTPException(status_code=401, detail="Session expired or invalid")
#     try:
#         return workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
#     except FileNotFoundError:
#         raise HTTPException(status_code=401, detail="Workspace no longer exists")

def get_me_workspace(request: Request, db: Session = Depends(get_db)):
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        raise HTTPException(status_code=401, detail="not joined")
    sess = db.execute(select(StudentSession).where(StudentSession.session_id == sid)).scalar_one_or_none()
    if not sess or sess.expires_at <= now_utc_naive():
        raise HTTPException(status_code=401, detail="Session expired")

    slot = db.get(WorkspaceSlot, sess.workspace_slot_id)
    if not slot:
        raise HTTPException(status_code=401, detail="Workspace missing") 
    
    return workspace.ensure_workspace_exists(RUNS_DIR, slot.ws_id)

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

@app.post("/debug/join")
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

@app.get("/me/active")
def me_active(ws: workspace.Workspace = Depends(get_me_workspace)):
    lab = workspace.get_active_lab(ws)
    return {"workspace_id": ws.id, "active_lab": lab}

@app.get("/me/nodes/{lab_name}")
def me_nodes(lab_name: str, ws: workspace.Workspace = Depends(get_me_workspace)):
    ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
    if not ws_yaml.exists():
        raise HTTPException(status_code=404, detail="Workspace topology not found")
    
    data = yaml.safe_load(ws_yaml.read_text(encoding="utf-8")) or {}
    nodes = list(((data.get("topology") or {}).get("nodes") or {}).keys())
    return {"workspace_id": ws.id, "lab_name": lab_name, "nodes": nodes}

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
    active = workspace.get_active_lab(ws)
    if active and active != lab_name:
        raise HTTPException(status_code=409, detail=f"Another lab is active: {active}. Please destroy it first.")

    tpl = workspace.resolve_template_yaml(LABS_DIR, lab_name)
    clab_name = workspace.make_clab_name(ws, tpl.stem)
    ws_yaml = workspace.ensure_topology_in_workspace(tpl, ws, clab_name=clab_name)

    r = clab.deploy(ws_yaml, reconfigure=True)

    if r.return_code == 0:
        workspace.set_active_lab(ws, lab_name)

    return {"workspace_id": ws.id, "clab_name": clab_name, **r.__dict__}

@app.post("/me/destroy/{lab_name}")
def me_destroy(lab_name: str, ws: workspace.Workspace = Depends(get_me_workspace)):
    ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
    if not ws_yaml.exists():
        raise HTTPException(status_code=404, detail=f"Workspace topology not found: {ws_yaml}")

    r = clab.destroy(ws_yaml, cleanup=True)

    if r.return_code == 0:
        workspace.clear_active_lab(ws)
    return {"workspace_id": ws.id, **r.__dict__}

@app.post("/me/reset/{lab_name}")
def me_reset(lab_name: str, ws: workspace.Workspace = Depends(get_me_workspace)):
    active = workspace.get_active_lab(ws)
    if active and active != lab_name:
        raise HTTPException(status_code=409, detail=f"Another lab is active: {active}. Please destroy it first.")

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

@app.post("/logout")
def logout(response: Response):
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return {"ok": True}

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

@app.delete("/workspaces/{ws_id}")
def cleanup_workspace(ws_id: str):
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        shutil.rmtree(ws.path)
        return {"message": f"Workspace {ws_id} deleted successfully."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

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

# @app.post("/workspaces/{ws_id}/deploy/{lab_name}")
# def deploy(ws_id: str, lab_name: str):
#     """
#     Deploy a lab into the given workspace.

#     Steps:
#     - verify workspace exists
#     - resolve the requested template YAML
#     - copy template YAML into the workspace
#     - deploy via containerlab using a deterministic lab name
#     """
#     try:
#         ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
#         tpl = workspace.resolve_template_yaml(LABS_DIR, lab_name)

#         clab_name = workspace.make_clab_name(ws, tpl.stem)
#         ws_yaml = workspace.ensure_topology_in_workspace(tpl, ws, clab_name=clab_name)
#         r = clab.deploy(ws_yaml, reconfigure=True)

#         return {"workspace_id": ws.id, "clab_name": clab_name, **r.__dict__}
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))


# @app.post("/workspaces/{ws_id}/destroy/{lab_name}")
# def destroy(ws_id: str, lab_name: str):
#     """
#     Destroy a lab deployment within the workspace.

#     This expects the workspace copy of the topology YAML to exist, ensuring
#     users do not destroy deployments using incorrect file paths.
#     """
#     try:
#         ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
#         ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
#         if not ws_yaml.exists():
#             raise FileNotFoundError(f"Workspace topology not found: {ws_yaml}")

#         r = clab.destroy(ws_yaml, cleanup=True)

#         return {"workspace_id": ws.id, **r.__dict__}
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))


# @app.get("/workspaces/{ws_id}/inspect/{lab_name}")
# def inspect(ws_id: str, lab_name: str):
#     """
#     Inspect a deployed lab within the workspace.

#     The response includes the raw stdout/stderr from `clab inspect`, which can be
#     parsed by the client if needed.
#     """
#     try:
#         ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
#         ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
#         if not ws_yaml.exists():
#             raise FileNotFoundError(f"Workspace topology not found: {ws_yaml}")

#         r = clab.inspect(ws_yaml)
#         return {"workspace_id": ws.id, **r.__dict__}
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))

# @app.get("/workspaces/{ws_id}/status/{lab_name}")
# def status(ws_id: str, lab_name: str):
#     try:
#         ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
#         ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
        
#         yaml_present = ws_yaml.exists()

#         if not yaml_present:
#             return {
#                 "workspace_id": ws.id,
#                 "lab_name": lab_name,
#                 "yaml_present": False,
#                 "running": False,
#                 "inspect": None
#             }
        
#         r = clab.inspect(ws_yaml)
#         running = (r.return_code == 0)
#         return {
#             "workspace_id": ws.id,
#             "lab_name": lab_name,
#             "yaml_present": True,
#             "running": running,
#             "inspect": {
#                 "return_code": r.return_code,
#                 "stdout": r.stdout,
#                 "stderr": r.stderr
#             }
#         }
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))

# @app.post("/workspaces/{ws_id}/reset/{lab_name}")
# def reset(ws_id: str, lab_name: str):
#     try:
#         ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
#         tpl = workspace.resolve_template_yaml(LABS_DIR, lab_name)
#         clab_name = workspace.make_clab_name(ws, tpl.stem)
#         ws_yaml = workspace.ensure_topology_in_workspace(tpl, ws, clab_name=clab_name)

#         r = clab.deploy(ws_yaml, reconfigure=True)
#         return {
#             "workspace_id": ws.id,
#             "clab_name": clab_name,
#             **r.__dict__
#         }
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))