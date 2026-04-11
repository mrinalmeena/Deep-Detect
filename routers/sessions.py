"""
routers/sessions.py — Session Endpoints
──────────────────────────────────────────
ENDPOINTS IN THIS FILE:
  POST /v1/session/start              → Create a new session
  GET  /v1/session/{session_id}       → Get session status
  DELETE /v1/session/{session_id}     → Delete session + purge video

WHAT IS A SESSION?
  A session is the container for one identity verification attempt.
  Flow:
    1. Client calls POST /v1/session/start
    2. Gets back session_id + upload_token
    3. Uses upload_token to upload video via POST /v1/verify/video
    4. Polls GET /v1/verify/status/{job_id} for result
    5. (Optional) GET /v1/session/{session_id} to see full session info
    6. DELETE /v1/session/{session_id} when done (data hygiene)
"""

from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid

from schemas import SessionStartRequest, SessionStartResponse, SessionStatusResponse, VerificationStatus
from database import get_db, SessionRecord

router = APIRouter()


# ════════════════════════════════════════════════════════
# POST /v1/session/start
# ════════════════════════════════════════════════════════

@router.post(
    "/start",
    response_model=SessionStartResponse,
    status_code=201,                        # 201 = Created (not 200)
    summary="Start a verification session",
    description="""
    Creates a new session and returns a short-lived upload token.
    
    **Step 1 of the verification flow.**
    
    Store the `session_id` — you'll need it to fetch the final report.
    Use the `upload_token` to authorise your video upload.
    The token expires in 15 minutes.
    """
)
async def start_session(
    body: SessionStartRequest,              # Validated automatically by Pydantic
    request: Request,                       # Gives access to request.state (customer_id)
    db: Session = Depends(get_db)          # DB session injected by FastAPI
):
    """
    HOW Depends(get_db) WORKS:
    FastAPI calls get_db() before calling this function,
    injects the db session as a parameter, and closes it
    after the function returns. You never manage the connection manually.
    """

    # ── Generate IDs ─────────────────────────────────────
    session_id   = f"sess_{uuid.uuid4().hex[:12]}"
    upload_token = f"tok_{uuid.uuid4().hex[:20]}"
    expires_at   = datetime.utcnow() + timedelta(minutes=15)

    # ── Save to database ──────────────────────────────────
    session = SessionRecord(
        session_id   = session_id,
        customer_id  = request.state.customer_id,   # Set by auth middleware
        user_ref     = body.user_ref,
        upload_token = upload_token,
        callback_url = body.callback_url,
        metadata_    = body.metadata,
        status       = VerificationStatus.PENDING,
        expires_at   = expires_at
    )
    db.add(session)
    db.commit()

    # ── Return response ───────────────────────────────────
    return SessionStartResponse(
        session_id   = session_id,
        upload_token = upload_token,
        expires_at   = expires_at,
        status       = VerificationStatus.PENDING
    )


# ════════════════════════════════════════════════════════
# GET /v1/session/{session_id}
# ════════════════════════════════════════════════════════

@router.get(
    "/{session_id}",
    response_model=SessionStatusResponse,
    summary="Get session details",
    description="Returns the current state of a session and its metadata."
)
async def get_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    # ── Look up session ───────────────────────────────────
    session = db.query(SessionRecord).filter(
        SessionRecord.session_id  == session_id,
        SessionRecord.customer_id == request.state.customer_id   # Security: own sessions only
    ).first()

    if not session:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": f"No session found with ID {session_id}.",
                "status_code": 404
            }
        )

    return SessionStatusResponse(
        session_id   = session.session_id,
        user_ref     = session.user_ref,
        status       = session.status,
        created_at   = session.created_at,
        expires_at   = session.expires_at,
        callback_url = session.callback_url,
        metadata     = session.metadata_
    )


# ════════════════════════════════════════════════════════
# DELETE /v1/session/{session_id}
# ════════════════════════════════════════════════════════

@router.delete(
    "/{session_id}",
    status_code=200,
    summary="Delete session and purge video data",
    description="""
    Permanently deletes the session record and any stored video.
    Use this to comply with GDPR Article 17 (right to erasure).
    This action cannot be undone.
    """
)
async def delete_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    session = db.query(SessionRecord).filter(
        SessionRecord.session_id  == session_id,
        SessionRecord.customer_id == request.state.customer_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=404,
            detail={"error": "session_not_found", "message": "Session not found.", "status_code": 404}
        )

    # In production: also delete video from S3 here
    # s3_client.delete_object(Bucket=BUCKET, Key=session.video_path)

    db.delete(session)
    db.commit()

    return {
        "message": f"Session {session_id} deleted successfully.",
        "session_id": session_id
    }