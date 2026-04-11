"""
schemas.py — Pydantic Models (Request & Response Shapes)
─────────────────────────────────────────────────────────
Pydantic does two things for us:
  1. VALIDATES incoming request data automatically
  2. SERIALIZES outgoing response data to clean JSON

Every endpoint uses these models — if data doesn't match
the schema, FastAPI returns a 422 error automatically.
No manual validation code needed.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


# ════════════════════════════════════════════════════════
# ENUMS  — fixed set of allowed string values
# ════════════════════════════════════════════════════════

class VerificationStatus(str, Enum):
    """Lifecycle of a verification job."""
    PENDING    = "pending"      # Job queued, not started
    PROCESSING = "processing"   # AI pipeline running
    COMPLETED  = "completed"    # Done, result available
    FAILED     = "failed"       # Error occurred


class Verdict(str, Enum):
    """Final liveness decision."""
    REAL         = "REAL"
    FAKE         = "FAKE"
    INCONCLUSIVE = "INCONCLUSIVE"  # Not enough data


# ════════════════════════════════════════════════════════
# SESSION SCHEMAS
# ════════════════════════════════════════════════════════

class SessionStartRequest(BaseModel):
    """
    Body sent when starting a new verification session.
    
    Example:
        POST /v1/session/start
        {
            "user_ref": "user_abc123",
            "callback_url": "https://yourapp.com/webhook"
        }
    """
    user_ref: str = Field(
        ...,                            # ... means REQUIRED
        min_length=1,
        max_length=128,
        description="Your internal user ID or reference.",
        example="user_abc123"
    )
    callback_url: Optional[str] = Field(
        None,
        description="Webhook URL — we POST the result here when done.",
        example="https://yourapp.com/webhooks/deepfake"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Any extra key-value pairs you want attached to this session.",
        example={"department": "engineering", "job_id": "jb_456"}
    )


class SessionStartResponse(BaseModel):
    """
    Returned immediately after POST /v1/session/start
    
    The client stores session_id and upload_token, then
    uses upload_token to call POST /v1/verify/video.
    """
    session_id: str = Field(
        ...,
        description="Unique session identifier. Keep this — you'll need it.",
        example="sess_a1b2c3d4"
    )
    upload_token: str = Field(
        ...,
        description="Short-lived token (15 min) to authorise the video upload.",
        example="tok_xyz987"
    )
    expires_at: datetime = Field(
        ...,
        description="When the upload_token expires. After this, start a new session.",
        example="2025-09-14T11:00:00Z"
    )
    status: VerificationStatus = Field(
        default=VerificationStatus.PENDING
    )


class SessionStatusResponse(BaseModel):
    """Response for GET /v1/session/{session_id}"""
    session_id: str
    user_ref: str
    status: VerificationStatus
    created_at: datetime
    expires_at: datetime
    callback_url: Optional[str]
    metadata: Optional[dict]


# ════════════════════════════════════════════════════════
# VERIFICATION SCHEMAS
# ════════════════════════════════════════════════════════

class DetectionSignals(BaseModel):
    """
    All the signals our AI pipeline extracts from the video.
    These are the raw measurements — the verdict is derived from these.
    """
    blink_count: int = Field(
        ...,
        ge=0,                           # ge = greater than or equal to 0
        description="Number of blinks detected in the video clip.",
        example=4
    )
    avg_ear: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Average Eye Aspect Ratio across all frames. ~0.3 = natural open eye.",
        example=0.312
    )
    min_ear: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Minimum EAR observed — how closed the eye got at peak blink.",
        example=0.089
    )
    head_yaw_deg: float = Field(
        ...,
        description="Head rotation left/right in degrees. 0 = straight ahead.",
        example=12.4
    )
    head_pitch_deg: float = Field(
        ...,
        description="Head tilt up/down in degrees.",
        example=-3.1
    )
    texture_score: float = Field(
        ...,
        ge=0.0,
        description="Laplacian variance of skin texture. Low = suspiciously smooth (possible deepfake).",
        example=28.6
    )
    motion_entropy: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Optical flow variance — real faces have natural micro-movements.",
        example=0.74
    )
    frames_analyzed: int = Field(
        ...,
        ge=0,
        description="Total number of frames the AI processed.",
        example=87
    )


class VerifyVideoResponse(BaseModel):
    """
    Returned immediately after POST /v1/verify/video.
    
    Processing is ASYNC — you get a job_id back right away.
    Poll GET /v1/verify/status/{job_id} for the result.
    
    Why async? Video processing takes 1-5 seconds. Holding
    the HTTP connection open that long is wasteful and doesn't
    scale. The queue handles it.
    """
    job_id: str = Field(
        ...,
        description="Use this to poll for your result.",
        example="job_a1b2c3"
    )
    session_id: str = Field(example="sess_a1b2c3d4")
    status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Will be 'pending' initially. Poll /status/{job_id} until 'completed'."
    )
    message: str = Field(
        default="Video received. Processing started.",
        description="Human-readable status message."
    )
    estimated_seconds: int = Field(
        default=3,
        description="Estimated processing time in seconds.",
        example=3
    )


class VerifyStatusResponse(BaseModel):
    """
    Full result — returned when GET /v1/verify/status/{job_id}
    returns status='completed'.
    
    If status is still 'pending' or 'processing', verdict
    and signals will be null — keep polling.
    """
    job_id: str
    session_id: str
    status: VerificationStatus

    # These are null until processing completes
    verdict: Optional[Verdict] = Field(
        None,
        description="REAL | FAKE | INCONCLUSIVE — null while processing.",
        example="REAL"
    )
    confidence: Optional[int] = Field(
        None,
        ge=0, le=100,
        description="0-100 confidence that the verdict is correct.",
        example=87
    )
    authenticity_score: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Normalised score: 1.0 = certainly real, 0.0 = certainly fake.",
        example=0.87
    )
    signals: Optional[DetectionSignals] = Field(
        None,
        description="Raw AI measurements. Null until processing completes."
    )
    processing_ms: Optional[int] = Field(
        None,
        description="How long AI processing took in milliseconds.",
        example=1240
    )
    created_at: datetime
    completed_at: Optional[datetime] = None
    expires_at: datetime


# ════════════════════════════════════════════════════════
# REPORT SCHEMA
# ════════════════════════════════════════════════════════

class ReportResponse(BaseModel):
    """
    Full audit report — returned by GET /v1/report/{session_id}
    
    This is the complete record for compliance/audit purposes.
    Includes everything from VerifyStatusResponse plus
    frame-level timeline data.
    """
    session_id: str
    user_ref: str
    job_id: Optional[str]

    # Final verdict
    verdict: Optional[Verdict]
    confidence: Optional[int]
    authenticity_score: Optional[float]

    # Raw signals
    signals: Optional[DetectionSignals]

    # Full frame-by-frame EAR timeline (for audit trail)
    ear_timeline: Optional[list[float]] = Field(
        None,
        description="EAR value for each analyzed frame, in order.",
        example=[0.31, 0.30, 0.09, 0.08, 0.28, 0.32]
    )

    # Metadata
    status: VerificationStatus
    processing_ms: Optional[int]
    frames_analyzed: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]


# ════════════════════════════════════════════════════════
# ERROR SCHEMA — consistent error format across all endpoints
# ════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """
    Every error in the API returns this shape.
    Makes it easy for clients to handle errors uniformly.
    
    Example:
        {
            "error": "session_not_found",
            "message": "No session with ID sess_xyz exists.",
            "status_code": 404
        }
    """
    error: str = Field(description="Machine-readable error code.", example="session_not_found")
    message: str = Field(description="Human-readable explanation.", example="Session not found.")
    status_code: int = Field(example=404)