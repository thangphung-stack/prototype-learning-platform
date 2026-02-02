from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base

class Lecturer(Base):
    __tablename__ = "lecturers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)

class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lecturer_id: Mapped[int] = mapped_column(ForeignKey("lecturers.id"))
    join_code: Mapped[str] = mapped_column(String, unique=True, index=True)
    starts_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    seats: Mapped[int] = mapped_column(Integer)

    lecturer = relationship("Lecturer")

    workspaces = relationship("WorkspaceSlot", back_populates="booking")

class WorkspaceSlot(Base):
    __tablename__ = "workspace_slots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(ForeignKey("bookings.id"), index=True)
    ws_id: Mapped[str] = mapped_column(String, index=True)

    assigned_session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    booking = relationship("Booking", back_populates="workspaces")

class StudentSession(Base):
    __tablename__ = "student_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    workspace_slot_id: Mapped[int] = mapped_column(ForeignKey("workspace_slots.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    expires_at: Mapped[datetime] = mapped_column(DateTime)

    __table_args__ = (
        UniqueConstraint("workspace_slot_id", name="uq_session_workspace_slot"),
    )