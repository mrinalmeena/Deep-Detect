"""
routers/reports.py — Report Endpoint
──────────────────────────────────────
ENDPOINTS IN THIS FILE:
  GET /v1/report/{session_id} → Full audit report for a session

WHY A SEPARATE REPORT ENDPOINT?
  /verify/status gives you the result for one specific job.
  /report gives you the FULL picture for an entire session —
  including the EAR timeline, frame count, and all metadata
  in one response suitable for compliance/audit logging.
"""

from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session

from schemas import ReportResponse, DetectionSignals, VerificationStatus
from database import get_db, SessionRecord, VerificationRecord

router = APIRouter()


# ════════════════════════════════════════════════════════
# GET /v1/report/{session_id}
# ════════════════════════════════════════════════════════

@router.get(
    "/{session_id}",
    response_model=ReportResponse,
    summary="Get full audit report for a session",
    description="""
    Returns the complete verification report for a session.
    
    Includes:
    - Final verdict and confidence score
    - All raw detection signals (EAR, head pose, texture, etc.)
    - Frame-by-frame EAR timeline for audit trail
    - Processing metadata
    
    Use this endpoint for:
    - Compliance and audit logging
    - Storing in your own database
    - Displaying results to your end user
    
    Reports are available for 24 hours after creation.
    """
)
async def get_report(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    # ── 1. Fetch session ──────────────────────────────────────────
    session = db.query(SessionRecord).filter(
        SessionRecord.session_id  == session_id,
        SessionRecord.customer_id == request.state.customer_id  # own sessions only
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail={
            "error": "session_not_found",
            "message": f"Session {session_id} not found.",
            "status_code": 404
        })

    # ── 2. Fetch latest verification job for this session ─────────
    job = db.query(VerificationRecord).filter(
        VerificationRecord.session_id == session_id
    ).order_by(VerificationRecord.created_at.desc()).first()

    # ── 3. Build signals object if available ─────────────────────
    signals = None
    if job and job.signals:
        signals = DetectionSignals(**job.signals)

    # ── 4. Return full report ─────────────────────────────────────
    return ReportResponse(
        session_id         = session.session_id,
        user_ref           = session.user_ref,
        job_id             = job.job_id if job else None,
        verdict            = job.verdict if job else None,
        confidence         = job.confidence if job else None,
        authenticity_score = job.authenticity_score if job else None,
        signals            = signals,
        ear_timeline       = job.ear_timeline if job else None,
        status             = job.status if job else VerificationStatus.PENDING,
        processing_ms      = job.processing_ms if job else None,
        frames_analyzed    = job.frames_analyzed if job else None,
        created_at         = session.created_at,
        completed_at       = job.completed_at if job else None
    )