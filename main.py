"""
main.py — DeepfakeID FastAPI Application Entry Point
─────────────────────────────────────────────────────
Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then visit:
    http://localhost:8000/docs   ← Auto-generated Swagger UI
    http://localhost:8000/redoc  ← ReDoc UI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import sessions, verify, reports
from middleware.auth import APIKeyMiddleware
from database import create_tables

# ── Create FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title="DeepfakeID API",
    description="Real-time liveness detection API for KYC, hiring, and onboarding.",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# ── CORS — allow your frontend domain in production ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Change to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Key auth middleware (applied to all routes) ───────────────
app.add_middleware(APIKeyMiddleware)

# ── Include routers (each router = one file of related endpoints) ─
app.include_router(sessions.router, prefix="/v1/session",  tags=["Sessions"])
app.include_router(verify.router,   prefix="/v1/verify",   tags=["Verification"])
app.include_router(reports.router,  prefix="/v1/report",   tags=["Reports"])

# ── Create DB tables on startup ───────────────────────────────────
@app.on_event("startup")
async def startup():
    create_tables()

# ── Health check (no auth required) ──────────────────────────────
@app.get("/health", tags=["System"], include_in_schema=True)
async def health():
    return {"status": "ok", "service": "DeepfakeID API", "version": "1.0.0"}