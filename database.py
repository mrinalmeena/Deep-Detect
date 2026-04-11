"""
database.py — Database Layer
──────────────────────────────
Uses SQLite for development (zero setup).
Swap DATABASE_URL in .env for PostgreSQL in production:
    DATABASE_URL=postgresql://user:pass@localhost/deepfakeid

HOW SQLAlchemy WORKS:
  - engine    = the database connection
  - SessionLocal = a factory that creates DB sessions
  - Base      = parent class all your table models inherit from

In production:
  pip install psycopg2-binary
  Set DATABASE_URL=postgresql://... in your .env
  Everything else stays the same.
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import os
import uuid

# ── Database URL ──────────────────────────────────────────────────
# Development: SQLite file (created automatically)
# Production:  set DATABASE_URL env var to your Postgres URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./deepfakeid.db")

engine = create_engine(
    DATABASE_URL,
    # check_same_thread only needed for SQLite
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ════════════════════════════════════════════════════════
# TABLE MODELS
# Each class = one table in the database
# ════════════════════════════════════════════════════════

class APIKeyRecord(Base):
    """
    Stores hashed API keys and their associated customer.
    
    NEVER store the raw key — only the SHA-256 hash.
    The raw key is shown to the customer ONCE at creation.
    """
    __tablename__ = "api_keys"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key_hash    = Column(String, unique=True, nullable=False, index=True)
    customer_id = Column(String, nullable=False)
    plan        = Column(String, default="starter")   # starter | growth | scale | enterprise
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)


class SessionRecord(Base):
    """
    One row per verification session.
    A session is created first, then a video is attached to it.
    """
    __tablename__ = "sessions"

    session_id   = Column(String, primary_key=True)
    customer_id  = Column(String, nullable=False, index=True)
    user_ref     = Column(String, nullable=False)       # customer's own user ID
    upload_token = Column(String, unique=True)
    callback_url = Column(String, nullable=True)
    metadata_    = Column(JSON,   nullable=True)
    status       = Column(String, default="pending")
    created_at   = Column(DateTime, default=datetime.utcnow)
    expires_at   = Column(DateTime)


class VerificationRecord(Base):
    """
    One row per video verification job.
    Linked to a session via session_id.
    """
    __tablename__ = "verifications"

    job_id              = Column(String, primary_key=True)
    session_id          = Column(String, nullable=False, index=True)
    status              = Column(String, default="pending")
    verdict             = Column(String, nullable=True)
    confidence          = Column(Integer, nullable=True)
    authenticity_score  = Column(Float,   nullable=True)

    # Detection signals stored as JSON
    signals             = Column(JSON, nullable=True)
    ear_timeline        = Column(JSON, nullable=True)   # list of floats

    processing_ms       = Column(Integer, nullable=True)
    frames_analyzed     = Column(Integer, nullable=True)
    video_path          = Column(String,  nullable=True)  # S3 key or local path

    created_at          = Column(DateTime, default=datetime.utcnow)
    completed_at        = Column(DateTime, nullable=True)
    expires_at          = Column(DateTime)


# ════════════════════════════════════════════════════════
# HELPERS — functions used by routes and middleware
# ════════════════════════════════════════════════════════

def create_tables():
    """Create all tables. Called once on app startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Dependency injected into routes via FastAPI's Depends().
    
    Usage in a route:
        @router.get("/something")
        def my_route(db: Session = Depends(get_db)):
            ...
    
    The 'finally' block ensures the DB session is always closed,
    even if an exception occurs inside the route.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_api_key_from_db(key_hash: str) -> dict | None:
    """
    Look up an API key by its hash.
    Returns customer info dict or None if not found/inactive.
    Called by the auth middleware on every request.
    """
    db = SessionLocal()
    try:
        record = db.query(APIKeyRecord).filter(
            APIKeyRecord.key_hash == key_hash,
            APIKeyRecord.is_active == True
        ).first()

        if not record:
            return None

        return {
            "customer_id": record.customer_id,
            "plan":        record.plan
        }
    finally:
        db.close()


def seed_test_api_key():
    """
    Creates a test API key for development.
    Run this once after starting the server.
    
    Test key: test_key_12345
    Hash stored in DB: sha256("test_key_12345")
    
    Usage:
        from database import seed_test_api_key
        seed_test_api_key()
    """
    import hashlib
    db = SessionLocal()
    try:
        test_hash = hashlib.sha256("test_key_12345".encode()).hexdigest()
        existing = db.query(APIKeyRecord).filter(
            APIKeyRecord.key_hash == test_hash
        ).first()

        if not existing:
            record = APIKeyRecord(
                key_hash    = test_hash,
                customer_id = "customer_test_001",
                plan        = "growth"
            )
            db.add(record)
            db.commit()
            print("✓ Test API key seeded: test_key_12345")
        else:
            print("✓ Test API key already exists.")
    finally:
        db.close()