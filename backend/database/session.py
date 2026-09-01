"""
VeriDoc — Database Session & Configuration (Phase 9)

Manages SQLAlchemy engine, session lifecycle, and automatic database schema creation.
"""

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "veridoc.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"

# Resolve DATABASE_URL from environment or fallback to SQLite in ./data
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

# Ensure data directory exists for SQLite
if DATABASE_URL.startswith("sqlite:///"):
    db_file_str = DATABASE_URL.replace("sqlite:///", "")
    db_file_path = Path(db_file_str)
    db_file_path.parent.mkdir(parents=True, exist_ok=True)

# SQLite multi-thread connection argument
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def init_db(target_engine=None) -> None:
    """Creates all database tables if they do not already exist."""
    active_engine = target_engine or engine
    Base.metadata.create_all(bind=active_engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a transactional database session.
    Automatically closes session upon request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
