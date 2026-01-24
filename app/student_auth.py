"""
Student "join + session" helpers.

Idea:
- Lecturer creates a bunch of workspaces (one per student).
- We generate a join code that hands out those workspace IDs one-by-one.
- Once a student gets a workspace, we give them a session cookie so they don't
  need the join code again for every request.

In-memory.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
import secrets
from typing import Dict, Optional
from fastapi import Cookie, HTTPException
from app import workspace

SESSION_COOKIE_NAME = "student_session"

#info about a student session
@dataclass
class SessionInfo:
    ws_id: str
    expires_at: datetime

class JoinCodeStore:
    """
    Keeps join codes and hands out workspace IDs.

    A join code maps to a list of workspace IDs.
    Each time a student uses the code, they get the "next" workspace.
    """    
    def __init__(self) -> None:
        self._lock = Lock()
        # join_code -> (ws_ids, next_index)
        self._pool: Dict[str, Tuple[List[str], int]] = {} #join_code -> (ws_ids, next index)

    def create_code(self, ws_ids: List[str]) -> str:
        """
        Create a new join code for a set of workspaces.

        Note: token_urlsafe gives us a short-ish code that's annoying to guess.
        """
        code = secrets.token_urlsafe(6)
        with self._lock:
            self._pool[code] = (ws_ids, 0)
        return code 
    
    def assign_workspace(self, code: str) -> str:
        """
        Assign (and reserve) the next workspace for this join code.

        Raises:
            KeyError: if the code doesn't exist
            RuntimeError: if the code has no more workspaces left
        """
        with self._lock:
            if code not in self._pool:
                raise KeyError("Invalid join code")

            ws_ids, idx = self._pool[code]

            if idx >= len(ws_ids):
                raise RuntimeError("No workspaces left for this code")

            ws_id = ws_ids[idx]
            self._pool[code] = (ws_ids, idx + 1)
            return ws_id

class StudentSessionStore:
    """
    In-memory session storage.

    Maps: session_id -> (workspace_id + expiry timestamp)

    This is good enough for a prototype. If you later run multiple API replicas,
    you'd move this to Redis / DB so sessions are shared.
    """
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: Dict[str, SessionInfo] = {}
    def create_session(self, ws_id: str, ttl_minutes: int = 60) -> str:
        """Create a new session ID for a workspace and set its expiry."""
        session_id = secrets.token_urlsafe(24)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
        with self._lock:
            self._sessions[session_id] = SessionInfo(ws_id=ws_id, expires_at=expires_at)
        return session_id
    def get_ws_id(self, session_id: str) -> Optional[str]:
        """
        Return the workspace ID for this session, or None if invalid/expired.

        We also clean up expired sessions on read (cheap + simple).
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            info = self._sessions.get(session_id) 

            if not info: 
                return None

            if info.expires_at <= now:
                del self._sessions[session_id]
                return None
            return info.ws_id

def get_current_workspace(
    runs_dir: str,
    session_id: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME), #Cookie(...) tells fastapi:
) -> workspace.Workspace:
    """
    FastAPI dependency.
    - `Cookie(..., alias=...)` tells FastAPI to read the cookie by name.
    - If the cookie is missing, the student basically hasn't joined yet.
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="Not joined (missing session cookie)")
    # We don't have the session store instance here (it's wired in main.py),
    # so this function acts as a placeholder.
    raise RuntimeError("Use make_workspace_dependency(...) from main.py")
