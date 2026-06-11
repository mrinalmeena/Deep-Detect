# ── DeepDetect – Hugging Face Spaces (Docker SDK) ─────────────────────────────
FROM python:3.10-slim

# System deps for librosa / soundfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user 1000 — create home dir and app dir with correct perms
RUN useradd -m -u 1000 user
WORKDIR /app

# Install CPU-only PyTorch first (avoids pulling ~5 GB of CUDA libs)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements-deploy.txt ./requirements-deploy.txt
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Pre-download the model during build so it's baked into the image
# This avoids downloading 1.26 GB at every cold start
RUN python -c "from transformers import AutoModelForAudioClassification, AutoFeatureExtractor; \
    AutoFeatureExtractor.from_pretrained('garystafford/wav2vec2-deepfake-voice-detector'); \
    AutoModelForAudioClassification.from_pretrained('garystafford/wav2vec2-deepfake-voice-detector')"

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Give ownership to the non-root user
RUN chown -R user:user /app

USER user

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Launch the FastAPI server
CMD uvicorn backend.main:app --host 0.0.0.0 --port 7860 --timeout-keep-alive 120
