"""
routers/verify.py — Verification Endpoints
────────────────────────────────────────────
ENDPOINTS IN THIS FILE:
  POST /v1/verify/video               → Upload video, start AI processing
  GET  /v1/verify/status/{job_id}     → Poll for result

ASYNC PROCESSING EXPLAINED:
  Video processing takes 1–5 seconds. Two options:
  
  Option A (synchronous — bad at scale):
    Client uploads → server processes → client waits → result returned
    Problem: HTTP connection held open 5s × thousands of users = crash
  
  Option B (asynchronous — what we use):
    Client uploads → server queues job → immediately returns job_id
    Client polls /status/{job_id} every second until completed
    Result available when status = "completed"
  
  In production, the queue is RabbitMQ/SQS.
  For this MVP, we simulate async with a background thread.
"""

from fastapi import APIRouter, Request, Depends, UploadFile, File, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
import asyncio

from schemas import VerifyVideoResponse, VerifyStatusResponse, VerificationStatus, Verdict, DetectionSignals
from database import get_db, SessionRecord, VerificationRecord
from ai.pipeline import run_ai_pipeline

router = APIRouter()


# ════════════════════════════════════════════════════════
# POST /v1/verify/video
# ════════════════════════════════════════════════════════

@router.post(
    "/video",
    response_model=VerifyVideoResponse,
    status_code=202,                # 202 = Accepted (processing started, not done yet)
    summary="Upload video for liveness analysis",
    description="""
    Upload a video clip (MP4/WebM, max 30s) for deepfake analysis.
    
    **Step 2 of the verification flow.**
    
    Returns immediately with a `job_id`. The AI pipeline runs async.
    Poll `GET /v1/verify/status/{job_id}` every 1-2 seconds until
    `status` is `completed`.
    
    Include the `upload_token` from `/session/start` in the
    `X-Upload-Token` header.
    """
)
async def upload_video(
    request:    Request,
    session_id: str       = Form(..., description="The session_id from /session/start"),
    video:      UploadFile = File(..., description="Video file. MP4 or WebM. Max 30s."),
    db:         Session   = Depends(get_db)
):
    # ── 1. Validate session exists and belongs to this customer ──
    session = db.query(SessionRecord).filter(
        SessionRecord.session_id  == session_id,
        SessionRecord.customer_id == request.state.customer_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail={
            "error": "session_not_found",
            "message": f"Session {session_id} not found.",
            "status_code": 404
        })

    # ── 2. Check session hasn't expired ──────────────────────────
    if session.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail={
            "error": "session_expired",
            "message": "This session has expired. Start a new session.",
            "status_code": 410
        })

    # ── 3. Validate file type ─────────────────────────────────────
    allowed_types = {"video/mp4", "video/webm", "video/quicktime"}
    if video.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail={
            "error": "unsupported_media",
            "message": f"File type {video.content_type} not supported. Use MP4 or WebM.",
            "status_code": 415
        })

    # ── 4. Read video bytes ───────────────────────────────────────
    video_bytes = await video.read()

    # ── 5. Enforce size limit (50MB) ──────────────────────────────
    max_size = 50 * 1024 * 1024     # 50MB in bytes
    if len(video_bytes) > max_size:
        raise HTTPException(status_code=413, detail={
            "error": "file_too_large",
            "message": "Video must be under 50MB.",
            "status_code": 413
        })

    # ── 6. Create a job record in the database ────────────────────
    job_id     = f"job_{uuid.uuid4().hex[:12]}"
    expires_at = datetime.utcnow() + timedelta(hours=24)

    job = VerificationRecord(
        job_id     = job_id,
        session_id = session_id,
        status     = VerificationStatus.PENDING,
        expires_at = expires_at
    )
    db.add(job)

    # Update session status
    session.status = VerificationStatus.PROCESSING
    db.commit()

    # ── 7. Run AI pipeline in background ─────────────────────────
    # run_in_threadpool runs the sync AI code in a thread pool
    # so it doesn't block the async event loop.
    # In production, this would push to SQS/RabbitMQ instead.
    asyncio.create_task(
        _process_video_async(job_id, session_id, video_bytes)
    )

    # ── 8. Return immediately — client will poll for result ───────
    return VerifyVideoResponse(
        job_id             = job_id,
        session_id         = session_id,
        status             = VerificationStatus.PENDING,
        message            = "Video received. AI pipeline started.",
        estimated_seconds  = 3
    )


async def _process_video_async(job_id: str, session_id: str, video_bytes: bytes):
    """
    Background coroutine that runs the AI pipeline and saves results.
    
    In production this would be a separate worker service
    consuming from a message queue, not running in the same process.
    But for MVP this is fine.
    """
    db = None
    try:
        from database import SessionLocal
        db = SessionLocal()

        # Update status to processing
        job = db.query(VerificationRecord).filter(
            VerificationRecord.job_id == job_id
        ).first()
        if job:
            job.status = VerificationStatus.PROCESSING
            db.commit()

        # Run AI pipeline (your OpenCV/MediaPipe code lives here)
        # This runs in a thread so it doesn't block async
        result = await run_in_threadpool(run_ai_pipeline, video_bytes)

        # Save result
        if job:
            job.status             = VerificationStatus.COMPLETED
            job.verdict            = result["verdict"]
            job.confidence         = result["confidence"]
            job.authenticity_score = result["authenticity_score"]
            job.signals            = result["signals"]
            job.ear_timeline       = result["ear_timeline"]
            job.processing_ms      = result["processing_ms"]
            job.frames_analyzed    = result["signals"]["frames_analyzed"]
            job.completed_at       = datetime.utcnow()
            db.commit()

    except Exception as e:
        # Mark job as failed — client will see status=failed when polling
        if db:
            job = db.query(VerificationRecord).filter(
                VerificationRecord.job_id == job_id
            ).first()
            if job:
                job.status = VerificationStatus.FAILED
                db.commit()
        print(f"[ERROR] AI pipeline failed for job {job_id}: {e}")
    finally:
        if db:
            db.close()


# ════════════════════════════════════════════════════════
# GET /v1/verify/status/{job_id}
# ════════════════════════════════════════════════════════

@router.get(
    "/status/{job_id}",
    response_model=VerifyStatusResponse,
    summary="Poll for verification result",
    description="""
    Poll this endpoint every 1-2 seconds after uploading a video.
    
    **Step 3 of the verification flow.**
    
    When `status` changes to `completed`, read the `verdict`,
    `confidence`, and `signals` fields.
    
    Status lifecycle:
    - `pending` → job queued
    - `processing` → AI pipeline running
    - `completed` → result ready
    - `failed` → error, try again
    """
)
async def get_verification_status(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    # ── Find the job ──────────────────────────────────────
    job = db.query(VerificationRecord).filter(
        VerificationRecord.job_id == job_id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail={
            "error": "job_not_found",
            "message": f"No job found with ID {job_id}.",
            "status_code": 404
        })

    # ── Build signals object if processing is complete ────
    signals = None
    if job.signals:
        signals = DetectionSignals(**job.signals)

    return VerifyStatusResponse(
        job_id             = job.job_id,
        session_id         = job.session_id,
        status             = job.status,
        verdict            = job.verdict,
        confidence         = job.confidence,
        authenticity_score = job.authenticity_score,
        signals            = signals,
        processing_ms      = job.processing_ms,
        created_at         = job.created_at,
        completed_at       = job.completed_at,
        expires_at         = job.expires_at
    )