# ── DeepDetect – Docker deployment ───────────────────────────────────────────
FROM python:3.10-slim

# System deps for librosa / soundfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only torch to keep image small)
COPY requirements-deploy.txt ./requirements-deploy.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Railway sets PORT env var; default to 8000 for local
ENV PORT=8000
EXPOSE $PORT

# Launch the FastAPI server on $PORT
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
