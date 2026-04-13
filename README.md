# 🛡️ DeepDetect — Audio Deepfake Detector

> **Expose AI-generated voices with forensic precision.**  
> Upload audio or record live. DeepDetect analyzes speech patterns at the signal level using a fine-tuned Wav2Vec2 transformer.

---

## 🎯 What It Does

DeepDetect is a local, privacy-first tool that detects whether a voice is **human or AI-generated**. It combines a state-of-the-art deep learning model with explainable acoustic signal analysis — giving you not just a verdict, but the *evidence* behind it.

No API keys. No data sent to the cloud. Everything runs on your machine.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ **Live Recording** | Record directly from your microphone in-browser |
| 📁 **File Upload** | Supports WAV, MP3, FLAC, OGG, M4A |
| 🧠 **Wav2Vec2 Model** | Fine-tuned transformer for deepfake audio classification |
| 📊 **Signal Visualisation** | Waveform + Mel Spectrogram rendered for every clip |
| 🔍 **Evidence Table** | Pitch variability, spectral flatness, ZCR, energy dynamics & more |
| ⚠️ **Risk Classification** | LOW / MEDIUM / HIGH / CRITICAL confidence levels |
| 🔒 **100% Local** | No data leaves your machine |

---

## 🖥️ Demo

```
Upload audio → Backend analyzes → Get verdict + evidence
```

**Landing page** → `http://localhost:8000`  
**Detection tool** → `http://localhost:8000/deepdetect-app.html`

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip
- A microphone (for live recording)

### 1. Clone the repo
```bash
git clone https://github.com/mrinalmeena/Deep-Detect.git
cd Deep-Detect
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
cd backend
python main.py
```

### 5. Open in browser
```
http://localhost:8000
```

That's it. No extra setup needed.

---

## 📁 Project Structure

```
Deep-Detect/
├── backend/
│   └── main.py              # FastAPI server + ML inference
├── frontend/
│   ├── index.html           # Landing page
│   ├── deepdetect-app.html  # Detection tool
│   └── hero.png
├── sample_audio/            # Test audio clips
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

1. **Ingest** — Audio uploaded or recorded, resampled to 16kHz mono
2. **Decompose** — Waveform + Mel spectrogram extracted
3. **Classify** — Wav2Vec2 transformer runs inference
4. **Explain** — Acoustic metrics analyzed: pitch variability, spectral flatness, ZCR, RMS energy, spectral rolloff
5. **Verdict** — Real or Fake, with confidence score and risk level

---

## 🔬 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI + Uvicorn |
| **ML Model** | Wav2Vec2 (`garystafford/wav2vec2-deepfake-voice-detector`) |
| **Audio Processing** | Librosa, NumPy |
| **Visualisation** | Matplotlib |
| **Frontend** | Vanilla HTML/CSS/JS |
| **ML Framework** | PyTorch + HuggingFace Transformers |

---

## ⚡ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Analyze an audio file for deepfake detection |
| `GET` | `/health` | Health check |

### Example request
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_audio.wav"
```

### Example response
```json
{
  "is_fake": true,
  "prob_real": 0.107,
  "prob_fake": 0.893,
  "confidence": 0.893,
  "risk": "CRITICAL",
  "duration": 3.2,
  "evidence": [...]
}
```

---

## 🎵 Best Results

- Use clips between **2–13 seconds**
- **16kHz mono** WAV gives highest accuracy
- Works best with clean, single-speaker audio

---

## 📄 License

MIT License — free to use, modify, and distribute.

---
