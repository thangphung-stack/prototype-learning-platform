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
from fastapi.staticfiles import StaticFiles
import yaml
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import select, update, text
from datetime import datetime, timedelta, timezone
import secrets

from app.db import engine, SessionLocal
from app.models import Lecturer, Booking, WorkspaceSlot, StudentSession
from app import security
import json
from app.catalog import load_catalog, topics_modes
from dataclasses import dataclass
import threading
import time

app = FastAPI()

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

def ensure_booking_columns():
    with engine.begin() as conn:
        cols = conn.execute(text("PRAGMA table_info(bookings)")).fetchall()
        names = {c[1] for c in cols}

        def add(col_sql: str):
            conn.execute(text(col_sql))

        if "ram_mb_per_seat" not in names:
            add("ALTER TABLE bookings ADD COLUMN ram_mb_per_seat INTEGER NOT NULL DEFAULT 512")
        if "closed_at" not in names:
            add("ALTER TABLE bookings ADD COLUMN closed_at DATETIME")
        if "selection_type" not in names:
            add("ALTER TABLE bookings ADD COLUMN selection_type VARCHAR")
        if "selection_key" not in names:
            add("ALTER TABLE bookings ADD COLUMN selection_key VARCHAR")
        if "allowed_labs_json" not in names:
            add("ALTER TABLE bookings ADD COLUMN allowed_labs_json TEXT NOT NULL DEFAULT '[]'")

@app.on_event("startup")
def startup():
    from app.db import Base
    Base.metadata.create_all(bind=engine)
    ensure_booking_columns()

#######################------------------Teacher authentication------------------#######################
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

#Linux exposes system memory info in a file called /proc/meminfo, 1 of the lines is "MemTotal: 123456 kB"
def host_mem_total_mb() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
                kb = int(line.split()[1])
                return kb // 1024
    return 0

#We dont allow bookings to reserve 100% of ram, since OS, docker, fastapi etc. need ram, so we only reserve 80% of total RAM
def capacity_mb() -> int:
    return int(host_mem_total_mb() * 0.8)

#the total reserved ram for a booking
def booking_reserved_mb(seats: int, ram_mb_per_seat: int):
    return seats * ram_mb_per_seat

#the total reserved ram for a time window, most important math part
#Goal: given a time window (start, end), compute how much ram is already reserved by other bookings that overlap with this window. Worst case scenario
def reserved_mb_for_window(db: Session, start: datetime, end: datetime) -> int:
    stmt = (
        select(Booking.seats, Booking.ram_mb_per_seat)
        .where(
            Booking.closed_at.is_(None),
            Booking.starts_at.is_not(None),
            Booking.ends_at.is_not(None),
            Booking.starts_at < end,
            Booking.ends_at > start,
        )
    )
    rows = db.execute(stmt).all()
    return sum(seats * ram for seats, ram in rows)

def now_utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

#receives a datetime obj or none. DB stores these as UTC time but naive datetime
def utc_z(dt: datetime | None) -> str | None:
    return (dt.isoformat() + "Z") if dt else None

def parse_dt_to_utc_naive(s: str | None) -> datetime | None:
    """
    Accepts ISO strings like:
      - "2026-01-06T10:00:00"
      - "2026-01-06T10:00:00Z"
      - "2026-01-06T10:00:00+01:00"
    Stores as naive UTC.
    """
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

class RegisterRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    email_existing = db.execute(select(Lecturer).where(Lecturer.email == req.email)).scalar_one_or_none()
    if email_existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    
    lec = Lecturer(email=req.email, password_hash=security.hash_password(req.password))
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

#source for teacher labs page
@app.get("/teacher/catalog")
def teacher_catalog(db: Session = Depends(get_db), lec: Lecturer = Depends(get_current_lecturer)):
    catalog = load_catalog(LABS_DIR)
    modes = topics_modes(catalog)
    return {
        "labs": [
            {
                "id": x.id,
                "title": x.title,
                "description": x.description,
                "ram_mb": x.ram_mb,
                "topic": x.topic,
                "level": x.level,
            } for x in catalog
        ],
        "modes": modes
    }

@app.get("/teacher/capacity")
def teacher_capacity(
    starts_at: str, ends_at: str, ram_mb_per_seat: int, 
    db: Session = Depends(get_db), lec: Lecturer = Depends(get_current_lecturer)
):
    start = parse_dt_to_utc_naive(starts_at)
    end = parse_dt_to_utc_naive(ends_at)
    if not start or not end or end <= start:
        raise HTTPException(status_code=400, detail="Invalid time window")
    
    cap = capacity_mb()
    reserved = reserved_mb_for_window(db, start, end)
    available = max(0, cap - reserved)
    max_seats = available // max(1, ram_mb_per_seat)

    return {
        "capacity_mb": cap,
        "reserved_mb": reserved,
        "available_mb": available,
        "ram_mb_per_seat": ram_mb_per_seat,
        "max_seats": int(max_seats),
    }

class CreateBookingRequest(BaseModel):
    seats: int
    starts_at: str
    ends_at: str
    selection_type: str
    selection_key: str

def make_join_code() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz1234567890"
    return "".join(secrets.choice(alphabet) for _ in range(8))

@app.post("/teacher/bookings")
def create_booking(
    req: CreateBookingRequest,
    db : Session = Depends(get_db),
    lec: Lecturer = Depends(get_current_lecturer)
):
    seats = req.seats
    if seats <= 0 or seats > 500:
        raise HTTPException(status_code=400, detail="Invalid number of seats")
    
    start = parse_dt_to_utc_naive(req.starts_at)
    end = parse_dt_to_utc_naive(req.ends_at)
    if not start or not end or end <= start:
        raise HTTPException(status_code=400, detail="Invalid time window")
    
    catalog = load_catalog(LABS_DIR)

    allowed_labs: list[str] = []
    ram_mb_per_seat: int | None = None

    if req.selection_type == "mode":
        #selection_key = "topic:level"
        if ":" not in req.selection_key:
            raise HTTPException(status_code=400, detail="Invalid mode selection key")
        topic, level = req.selection_key.split(":", 1)
        labs_in_mode = [x for x in catalog if x.topic == topic and x.level == level]
        if not labs_in_mode:
            raise HTTPException(status_code=400, detail="Mode contains no labs")
        
        allowed_labs = [x.id for x in labs_in_mode]
        ram_mb_per_seat = max(x.ram_mb for x in labs_in_mode)
    
    elif req.selection_type == "max_lab":
        #selection_key = lab_id
        chosen = next((x for x in catalog if x.id == req.selection_key), None)
        if not chosen:
            raise HTTPException(status_code=400, detail="Unknown lab selection")
        
        allowed = [x for x in catalog if x.ram_mb <= chosen.ram_mb]
        allowed_labs = [x.id for x in allowed]
        ram_mb_per_seat = chosen.ram_mb
    
    else:
        raise HTTPException(status_code=400, detail="selection_type must be mode or max_lab")
    
    cap = capacity_mb()
    reserved = reserved_mb_for_window(db, start, end)
    available = max(0, cap - reserved)
    max_seats = available // max(1, ram_mb_per_seat)

    if seats > max_seats:
        raise HTTPException(
            status_code=400,
            detail=f"Seats exceeds max_seats for this window. max_seats={max_seats}",
        )

    for _ in range(20):
        join_code = make_join_code()
        exists = db.execute(select(Booking).where(Booking.join_code == join_code)).scalar_one_or_none()
        if not exists:
            break
    else:
        raise HTTPException(status_code=500, detail="Could not generate unique join code")    
    
    booking = Booking(
        lecturer_id=lec.id,
        join_code=join_code,
        seats=seats,
        starts_at=start,
        ends_at=end,
        ram_mb_per_seat=ram_mb_per_seat,
        closed_at=None,
        selection_type=req.selection_type,
        selection_key=req.selection_key,
        allowed_labs_json=json.dumps(allowed_labs),        
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
        "seats": seats,
        "starts_at": start.isoformat() + "Z",
        "ends_at": end.isoformat() + "Z",    
        "ram_mb_per_seat": ram_mb_per_seat,
        "allowed_labs": allowed_labs,
        "max_seats": int(max_seats),
        "join_url": "/ui/",
    }

def booking_status(b: Booking, now: datetime) -> str:
    if b.closed_at: return "closed"
    if b.starts_at and now < b.starts_at: return "upcoming"
    if b.ends_at and now > b.ends_at: return "past"
    return "active"

@app.get("/teacher/bookings")
def list_bookings(db: Session = Depends(get_db), lec: Lecturer = Depends(get_current_lecturer)):
    now = now_utc_naive()
    stmt = select(Booking).where(Booking.lecturer_id == lec.id).order_by(Booking.id.desc())
    bookings = db.execute(stmt).scalars().all()

    return {
        "bookings": [{
            "id": b.id,
            "join_code": b.join_code,
            "seats": b.seats,
            "starts_at": utc_z(b.starts_at),
            "ends_at": utc_z(b.ends_at),
            "ram_mb_per_seat": b.ram_mb_per_seat,
            "closed_at": utc_z(b.closed_at),
            "status": booking_status(b, now),
        } for b in bookings]
    }

@app.get("/teacher/bookings/{booking_id}")
def get_booking(booking_id: int, db: Session = Depends(get_db), lec: Lecturer = Depends(get_current_lecturer)):
    b = db.get(Booking, booking_id)
    if not b or b.lecturer_id != lec.id:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    assigned = db.execute(
        select(WorkspaceSlot).where(WorkspaceSlot.booking_id == b.id, WorkspaceSlot.assigned_at.is_not(None))).scalars().all()

    return {
        "id": b.id,
        "join_code": b.join_code,
        "seats": b.seats,
        "starts_at": utc_z(b.starts_at),
        "ends_at": utc_z(b.ends_at),
        "assigned_count": len(assigned),
    }

def close_booking_internal(db: Session, booking: Booking) -> dict:
    slots = db.execute(select(WorkspaceSlot).where(WorkspaceSlot.booking_id == booking.id)).scalars().all()

    destroyed_attempts = 0
    removed_dirs = 0

    for slot in slots:
        try:
            ws = workspace.ensure_workspace_exists(RUNS_DIR, slot.ws_id)
        except FileNotFoundError:
            continue

        # try destroy for ANY yml in workspace (safer than relying on .active_lab)
        for yml in ws.path.glob("*.yml"):
            lab = yml.stem
            clab_name = workspace.make_clab_name(ws, lab)
            clab.destroy(yml, name=clab_name, cleanup=True)
            destroyed_attempts += 1

        # delete workspace dir
        try:
            shutil.rmtree(ws.path)
            removed_dirs += 1
        except Exception:
            pass

        slot.assigned_at = None
        slot.assigned_session_id = None

    booking.closed_at = now_utc_naive()
    return {"workspaces_total": len(slots), "destroy_attempts": destroyed_attempts, "workspace_dirs_removed": removed_dirs}


@app.post("/teacher/bookings/{booking_id}/close")
def close_booking(booking_id: int, db: Session = Depends(get_db), lec: Lecturer = Depends(get_current_lecturer)):
    b = db.get(Booking, booking_id)
    if not b or b.lecturer_id != lec.id:
        raise HTTPException(status_code=404, detail="Booking not found")

    info = close_booking_internal(db, b)
    db.commit()

    return {"ok": True, "booking_id": booking_id, **info}


####################-----------------Students-----------------####################
def get_student_session_id(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)

@app.post("/join") 
def join(code: str, response: Response, request: Request, db: Session = Depends(get_db)):
    now = now_utc_naive()

    booking = db.execute(select(Booking).where(Booking.join_code == code)).scalar_one_or_none()
    if not booking:
        response.delete_cookie(SESSION_COOKIE, path="/")
        raise HTTPException(status_code=404, detail="Invalid join code")

    if booking.closed_at is not None:
        response.delete_cookie(SESSION_COOKIE, path="/")
        raise HTTPException(status_code=403, detail="Booking is closed")

    if booking.starts_at and now < booking.starts_at:
        response.delete_cookie(SESSION_COOKIE, path="/")
        raise HTTPException(status_code=403, detail="Booking/Class has not started yet")

    if booking.ends_at and now > booking.ends_at:
        response.delete_cookie(SESSION_COOKIE, path="/")
        raise HTTPException(status_code=403, detail="Booking/Class has ended")

    # Rejoin: reuse existing cookie ONLY if it belongs to this booking
    sid = get_student_session_id(request)
    if sid:
        sess = db.execute(select(StudentSession).where(StudentSession.session_id == sid)).scalar_one_or_none()
        if sess and sess.expires_at > now:
            slot = db.get(WorkspaceSlot, sess.workspace_slot_id)
            if slot and slot.booking_id == booking.id:
                return {"ok": True}
        response.delete_cookie(SESSION_COOKIE, path="/")

    # session expiry capped by booking end
    session_exp = now + timedelta(hours=6)
    if booking.ends_at:
        session_exp = min(session_exp, booking.ends_at)

    for _ in range(10):
        slot_id = db.execute(
            select(WorkspaceSlot.id)
            .where(
                WorkspaceSlot.booking_id == booking.id,
                WorkspaceSlot.assigned_at.is_(None),
            )
            .order_by(WorkspaceSlot.id.asc())
            .limit(1)
        ).scalar_one_or_none()

        if not slot_id:
            raise HTTPException(status_code=409, detail="No workspaces left for this booking")

        new_sid = secrets.token_hex(16)

        res = db.execute(
            update(WorkspaceSlot)
            .where(WorkspaceSlot.id == slot_id, WorkspaceSlot.assigned_at.is_(None))
            .values(assigned_at=now, assigned_session_id=new_sid)
        )

        if res.rowcount == 1:
            sess = StudentSession(
                session_id=new_sid,
                workspace_slot_id=slot_id,
                created_at=now,
                expires_at=session_exp,
            )
            db.add(sess)
            db.commit()

            response.set_cookie(
                key=SESSION_COOKIE,
                value=new_sid,
                httponly=True,
                samesite="lax",
                secure=False,  # MUST stay False on HTTP
                max_age=max(1, int((session_exp - now).total_seconds())),
                path="/",
            )
            return {"ok": True}

        db.rollback()

    raise HTTPException(status_code=503, detail="Please retry join (temporary contention)")

######################## ---- Student me endpoints ----############################

@dataclass
class MeContext:
    ws: workspace.Workspace
    booking: Booking
    slot: WorkspaceSlot
    sess: StudentSession

def get_me_context(request: Request, db: Session = Depends(get_db)) -> MeContext:
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        raise HTTPException(status_code=401, detail="not joined")
    
    now = now_utc_naive()

    sess = db.execute(select(StudentSession).where(StudentSession.session_id == sid)).scalar_one_or_none()
    if not sess or sess.expires_at <= now:
        raise HTTPException(status_code=401, detail="Session expired")
    
    slot = db.get(WorkspaceSlot, sess.workspace_slot_id)
    if not slot:
        raise HTTPException(status_code=401, detail="Workspace missing")
    
    booking = db.get(Booking, slot.booking_id)
    if not booking:
        raise HTTPException(status_code=401, detail="Booking missing")
    
    #hard policy
    if booking.closed_at is not None:
        raise HTTPException(status_code=403, detail="Booking is closed")
    if booking.starts_at and now < booking.starts_at:
        raise HTTPException(status_code=403, detail="Booking/Class has not started yet")
    if booking.ends_at and now > booking.ends_at:
        raise HTTPException(status_code=403, detail="Booking/Class has ended")
    
    ws = workspace.ensure_workspace_exists(RUNS_DIR, slot.ws_id)
    return MeContext(ws=ws, booking=booking, slot=slot, sess=sess)

@app.get("/me/guard")
def me_guard(ctx: MeContext = Depends(get_me_context)):
    return {"ok": True}

@app.get("/me/active")
def me_active(ctx: MeContext = Depends(get_me_context)):
    ws = ctx.ws
    lab = workspace.get_active_lab(ws)
    return {"workspace_id": ws.id, "active_lab": lab}

@app.get("/me/nodes/{lab_name}")
def me_nodes(lab_name: str, ctx: MeContext = Depends(get_me_context)):
    allowed = json.loads(ctx.booking.allowed_labs_json or "[]")
    if lab_name not in allowed:
        raise HTTPException(status_code=403, detail="Lab not allowed for this booking")
    
    ws = ctx.ws
    ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
    if not ws_yaml.exists():
        raise HTTPException(status_code=404, detail="Workspace topology not found")
    
    data = yaml.safe_load(ws_yaml.read_text(encoding="utf-8")) or {}
    nodes = list(((data.get("topology") or {}).get("nodes") or {}).keys())
    return {"workspace_id": ws.id, "lab_name": lab_name, "nodes": nodes}

@app.get("/me/labs")
def me_labs(ctx: MeContext = Depends(get_me_context)):
    ws = ctx.ws
    allowed = json.loads(ctx.booking.allowed_labs_json or "[]")

    template_map = {f.stem: f for f in LABS_DIR.glob("*.yml")}
    labs = []
    for lab_id in allowed:
        f = template_map.get(lab_id)
        if not f:
            continue
        labs.append({
            "id": lab_id,
            "filename": f.name,
            "in_workspace": (ws.path / f.name).exists(),
        })

    return {"workspace_id": ws.id, "labs": labs}

@app.post("/me/deploy/{lab_name}")
def me_deploy(lab_name: str, ctx: MeContext = Depends(get_me_context)):
    allowed = json.loads(ctx.booking.allowed_labs_json or "[]")
    if lab_name not in allowed:
        raise HTTPException(status_code=403, detail="Lab not allowed for this booking")
    
    ws = ctx.ws
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
def me_destroy(lab_name: str, ctx: MeContext = Depends(get_me_context)):
    allowed = json.loads(ctx.booking.allowed_labs_json or "[]")
    if lab_name not in allowed:
        raise HTTPException(status_code=403, detail="Lab not allowed for this booking")
    
    ws = ctx.ws
    ws_yaml = workspace.workspace_yaml_path(ws, lab_name)
    if not ws_yaml.exists():
        raise HTTPException(status_code=404, detail=f"Workspace topology not found: {ws_yaml}")

    r = clab.destroy(ws_yaml, cleanup=True)

    if r.return_code == 0:
        workspace.clear_active_lab(ws)
    return {"workspace_id": ws.id, **r.__dict__}

@app.post("/me/reset/{lab_name}")
def me_reset(lab_name: str, ctx: MeContext = Depends(get_me_context)):
    allowed = json.loads(ctx.booking.allowed_labs_json or "[]")
    if lab_name not in allowed:
        raise HTTPException(status_code=403, detail="Lab not allowed for this booking")
    
    ws = ctx.ws
    active = workspace.get_active_lab(ws)
    if active and active != lab_name:
        raise HTTPException(status_code=409, detail=f"Another lab is active: {active}. Please destroy it first.")

    tpl = workspace.resolve_template_yaml(LABS_DIR, lab_name)
    clab_name = workspace.make_clab_name(ws, tpl.stem)
    ws_yaml = workspace.ensure_topology_in_workspace(tpl, ws, clab_name=clab_name)
    r = clab.deploy(ws_yaml, reconfigure=True)
    return {"workspace_id": ws.id, "clab_name": clab_name, **r.__dict__}


@app.get("/me/status/{lab_name}")
def me_status(lab_name: str, ctx: MeContext = Depends(get_me_context)):
    allowed = json.loads(ctx.booking.allowed_labs_json or "[]")
    if lab_name not in allowed:
        raise HTTPException(status_code=403, detail="Lab not allowed for this booking")
    
    ws = ctx.ws
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
    response.delete_cookie(key=SESSION_COOKIE)
    return {"ok": True}

@app.get("/")
def home():
    return {"message": "hello from prototype-learning-platform"}

_sweeper_started = False

def sweeper_loop():
    while True:
        time.sleep(60)
        db = SessionLocal()
        try:
            now = now_utc_naive()
            expired = db.execute(
                select(Booking).where(
                    Booking.closed_at.is_(None),
                    Booking.ends_at.is_not(None),
                    Booking.ends_at < now
                )
            ).scalars().all()

            for b in expired:
                close_booking_internal(db, b)

            if expired:
                db.commit()
        finally:
            db.close()

@app.on_event("startup")
def start_sweeper():
    global _sweeper_started
    if _sweeper_started:
        return
    _sweeper_started = True
    threading.Thread(target=sweeper_loop, daemon=True).start()

@app.get("/list")
def list_labs():
    """Return all available lab templates found in the labs directory."""
    labs = [{"id": f.stem, "filename": f.name} for f in LABS_DIR.glob("*.yml")]
    return {"labs": labs}

@app.delete("/workspaces/{ws_id}")
def cleanup_workspace(ws_id: str):
    try:
        ws = workspace.ensure_workspace_exists(RUNS_DIR, ws_id)
        shutil.rmtree(ws.path)
        return {"message": f"Workspace {ws_id} deleted successfully."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))