# DeepfakeID — Real-Time Liveness Detection API

> **B2B SaaS API for detecting deepfake identity fraud during video verification.**  
> Built for KYC, remote hiring, onboarding, and exam proctoring platforms.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-MVP-yellow?style=flat-square)

---

##  What is DeepfakeID?

DeepfakeID is a **real-time liveness detection API** that protects identity verification workflows from deepfake attacks.

When someone joins a video call for KYC, a job interview, or an online exam — how do you know they're a real person and not a pre-recorded video or AI-generated face?

DeepfakeID answers that question in under 3 seconds.

It analyses a short video clip using computer vision and returns:
- `REAL` — live human detected
- `FAKE` — deepfake or spoofing attempt detected
- `INCONCLUSIVE` — not enough signal

---

## The Problem

Modern deepfake tools allow bad actors to:
- Loop pre-recorded videos during KYC checks
- Use AI-generated synthetic faces to open bank accounts
- Spoof identity in remote hiring interviews
- Bypass proctoring systems during online exams

Photo-based verification is no longer enough. **Liveness detection is the missing layer.**

---

##  The Solution

A drop-in REST API that any platform can integrate in minutes.

```
Your App → POST /v1/verify/video → DeepfakeID AI Pipeline → REAL / FAKE verdict
```

No hardware required. No SDK installation. Just an HTTP request.

---

## How It Works

DeepfakeID extracts multiple signals from each video frame:

| Signal | Method | Why It Matters |
|--------|--------|----------------|
| **Blink detection** | Eye Aspect Ratio (EAR) | Real eyes blink; static images don't |
| **Skin texture** | Laplacian variance | Real skin has micro-texture; GAN faces are smooth |
| **Head pose** | solvePnP (6-point model) | Real people move; deepfakes stay static |
| **Motion entropy** | Optical flow variance | Natural micro-movements vs. looped video |

These signals feed into a **weighted confidence scorer** that outputs a 0–100 score and a final verdict.

---

## Project Structure

```
deepfakeid_api/
│
├── main.py                  ← FastAPI app entry point
├── schemas.py               ← Pydantic request/response models
├── database.py              ← SQLAlchemy ORM + DB helpers
├── seed.py                  ← Creates test API key for development
├── requirements.txt
├── .env.example
│
├── middleware/
│   ├── __init__.py
│   └── auth.py              ← API key authentication middleware
│
├── routers/
│   ├── __init__.py
│   ├── sessions.py          ← POST /session/start, GET/DELETE /session/{id}
│   ├── verify.py            ← POST /verify/video, GET /verify/status/{job_id}
│   └── reports.py           ← GET /report/{session_id}
│
└── ai/
    ├── __init__.py
    └── pipeline.py          ← OpenCV + MediaPipe detection logic
```

---

##  Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/deepfakeid-api.git
cd deepfakeid-api/deepfakeid_api
```

### 2. Create a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate        
myenv\Scripts\activate           
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env

```

### 5. Create the database and seed a test API key

```bash
python seed.py
```

Output:
```
Creating tables...
✓ Test API key seeded: test_key_12345
```

### 6. Start the server

```bash
uvicorn main:app --reload --port 8000
```

### 7. Open the interactive API docs

```
http://localhost:8000/docs
```

---

##  Authentication

All endpoints require an API key in the request header:

```bash
X-API-Key: your_api_key_here
```

For local development, use the seeded test key:

```bash
X-API-Key: test_key_12345
```

---

## 📡 API Endpoints

### `POST /v1/session/start`
Start a new verification session.

```bash
curl -X POST http://localhost:8000/v1/session/start \
  -H "X-API-Key: test_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"user_ref": "user_abc123", "callback_url": "https://yourapp.com/webhook"}'
```

Response:
```json
{
  "session_id": "sess_a1b2c3d4",
  "upload_token": "tok_xyz987",
  "expires_at": "2025-09-14T11:00:00Z",
  "status": "pending"
}
```

---

### `POST /v1/verify/video`
Upload a video for analysis. Returns immediately with a `job_id`.

```bash
curl -X POST http://localhost:8000/v1/verify/video \
  -H "X-API-Key: test_key_12345" \
  -F "session_id=sess_a1b2c3d4" \
  -F "video=@/path/to/video.mp4"
```

Response:
```json
{
  "job_id": "job_a1b2c3",
  "session_id": "sess_a1b2c3d4",
  "status": "pending",
  "message": "Video received. AI pipeline started.",
  "estimated_seconds": 3
}
```

---

### `GET /v1/verify/status/{job_id}`
Poll for the result. Keep polling until `status` = `completed`.

```bash
curl http://localhost:8000/v1/verify/status/job_a1b2c3 \
  -H "X-API-Key: test_key_12345"
```

Response:
```json
{
  "job_id": "job_a1b2c3",
  "session_id": "sess_a1b2c3d4",
  "status": "completed",
  "verdict": "REAL",
  "confidence": 87,
  "authenticity_score": 0.87,
  "signals": {
    "blink_count": 4,
    "avg_ear": 0.312,
    "min_ear": 0.089,
    "head_yaw_deg": 12.4,
    "head_pitch_deg": -3.1,
    "texture_score": 28.6,
    "motion_entropy": 0.74,
    "frames_analyzed": 87
  },
  "processing_ms": 1240
}
```

---

### `GET /v1/session/{session_id}`
Get session details and current status.

### `DELETE /v1/session/{session_id}`
Delete session and purge video data (GDPR Article 17 compliance).

### `GET /v1/report/{session_id}`
Full audit report with frame-by-frame EAR timeline.

---

##  Verification Flow

```
1. POST /v1/session/start      → get session_id + upload_token
2. POST /v1/verify/video       → upload video, get job_id
3. GET  /v1/verify/status/{id} → poll every 1s until status=completed
4. GET  /v1/report/{id}        → fetch full audit report
5. DELETE /v1/session/{id}     → purge data when done
```

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Database | SQLAlchemy + SQLite (dev) / PostgreSQL (prod) |
| Computer Vision | OpenCV 4.9 |
| Face Landmarks | MediaPipe Face Mesh |
| Authentication | SHA-256 hashed API keys |
| Language | Python 3.10 |

---

## Roadmap

- [x] EAR-based blink detection
- [x] Head pose estimation
- [x] Skin texture analysis
- [x] Motion entropy (optical flow)
- [x] FastAPI REST backend
- [x] API key authentication
- [x] Async video processing
- [ ] CNN deepfake model (FaceForensics++)
- [ ] WebSocket real-time streaming endpoint
- [ ] Lip-sync detection
- [ ] Docker + docker-compose
- [ ] AWS deployment (ECS + SQS)
- [ ] Usage dashboard
- [ ] Stripe billing integration

---

##  Security

- API keys stored as SHA-256 hashes — raw keys never stored
- TLS required in production
- Video files deleted after 24 hours by default
- No biometric templates stored — only derived scores
- GDPR Article 17 compliant via DELETE endpoint

---

##  Target Customers

- **Fintech / Banking** — KYC onboarding, account opening
- **Hiring Platforms** — Remote interview identity verification
- **Edtech** — Online exam proctoring
- **Healthcare** — Telehealth patient verification
- **Government** — Digital identity verification

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

##  Built at VIHAAN Hackathon

> DeepfakeID was prototyped at VIHAAN using Python, OpenCV, and MediaPipe.  
> The vision: make liveness detection as easy to integrate as a payment API.