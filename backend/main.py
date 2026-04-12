import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ─── App Setup ───────────────────────────────────────────────────────────────────
app = FastAPI(title="DeepDetect API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model ─────────────────────────────────────────────────────────────────
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
print("Loading model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.")


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_waveform_png(waveform_np: np.ndarray, duration: float) -> str:
    fig, ax = plt.subplots(figsize=(14, 4))
    t = np.linspace(0, duration, len(waveform_np))
    ax.fill_between(t, waveform_np, alpha=0.22, color="#2dd4bf")
    ax.plot(t, waveform_np, color="#2dd4bf", linewidth=0.35)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.set_xlabel("Time (s)", color="#64748b", fontsize=10)
    ax.set_ylabel("Amplitude", color="#64748b", fontsize=10)
    ax.tick_params(colors="#475569", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#1e293b")
    ax.set_xlim(0, duration)
    return fig_to_base64(fig)


def make_spectrogram_png(waveform_np: np.ndarray, sr: int, duration: float) -> str:
    mel = librosa.feature.melspectrogram(y=waveform_np, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(14, 4))
    img = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="inferno",
                    extent=[0, duration, 0, 80])
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.set_xlabel("Time (s)", color="#64748b", fontsize=10)
    ax.set_ylabel("Mel Bin", color="#64748b", fontsize=10)
    ax.tick_params(colors="#475569", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#1e293b")
    cbar = fig.colorbar(img, ax=ax, format="%+.0f dB", pad=0.02)
    cbar.ax.tick_params(colors="#475569", labelsize=9)
    cbar.outline.set_edgecolor("#1e293b")
    return fig_to_base64(fig)


def compute_metrics(waveform_np: np.ndarray, sr: int = 16000) -> dict:
    m = {}
    pitches, mags = librosa.piptrack(y=waveform_np, sr=sr, fmin=50, fmax=400)
    pvals = []
    for t in range(pitches.shape[1]):
        idx = mags[:, t].argmax()
        p = pitches[idx, t]
        if p > 0:
            pvals.append(p)
    if len(pvals) > 2:
        pa = np.array(pvals)
        m["pitch_variability"] = float(np.std(pa) / (np.mean(pa) + 1e-8))
    else:
        m["pitch_variability"] = 0.0

    sf = librosa.feature.spectral_flatness(y=waveform_np)
    m["spectral_flatness"] = float(np.mean(sf))

    zcr = librosa.feature.zero_crossing_rate(waveform_np)
    m["zcr_std"] = float(np.std(zcr))

    rms = librosa.feature.rms(y=waveform_np)
    m["rms_variability"] = float(np.std(rms) / (np.mean(rms) + 1e-8))

    rolloff = librosa.feature.spectral_rolloff(y=waveform_np, sr=sr)
    m["rolloff_std"] = float(np.std(rolloff))

    return m


def build_evidence(metrics: dict) -> list:
    rows = []

    pv = metrics["pitch_variability"]
    if pv < 0.08:
        a, c = "Unnaturally steady", "suspect"
    elif pv > 0.25:
        a, c = "Natural variation", "normal"
    else:
        a, c = "Moderate", "neutral"
    rows.append({"metric": "Pitch Variability", "value": round(pv, 4), "pct": min(round(pv / 0.3 * 100), 100), "assessment": a, "tag": c})

    sf = metrics["spectral_flatness"]
    if sf > 0.05:
        a, c = "Synthetic texture", "suspect"
    elif sf < 0.01:
        a, c = "Natural spectrum", "normal"
    else:
        a, c = "Borderline", "neutral"
    rows.append({"metric": "Spectral Flatness", "value": round(sf, 5), "pct": min(round(sf / 0.08 * 100), 100), "assessment": a, "tag": c})

    rv = metrics["rms_variability"]
    if rv < 0.3:
        a, c = "Too uniform", "suspect"
    elif rv > 0.6:
        a, c = "Natural dynamics", "normal"
    else:
        a, c = "Moderate", "neutral"
    rows.append({"metric": "Energy Dynamics", "value": round(rv, 4), "pct": min(round(rv / 1.0 * 100), 100), "assessment": a, "tag": c})

    zs = metrics["zcr_std"]
    if zs < 0.01:
        a, c = "Low variation", "suspect"
    else:
        a, c = "Normal variation", "normal"
    rows.append({"metric": "ZCR Variation", "value": round(zs, 5), "pct": min(round(zs / 0.05 * 100), 100), "assessment": a, "tag": c})

    ro = metrics["rolloff_std"]
    if ro < 200:
        a, c = "Rigid spectrum", "suspect"
    else:
        a, c = "Dynamic spectrum", "normal"
    rows.append({"metric": "Spectral Rolloff Std", "value": f"{ro:.0f} Hz", "pct": min(round(ro / 1000 * 100), 100), "assessment": a, "tag": c})

    return rows


# ─── Endpoint ────────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    waveform_np, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    duration = round(len(waveform_np) / 16000, 2)

    # Model inference with temperature scaling to soften extreme confidence
    TEMPERATURE = 2.5
    inputs = feature_extractor(waveform_np, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits / TEMPERATURE, dim=-1)

    prob_real = round(probs[0][0].item(), 4)
    prob_fake = round(probs[0][1].item(), 4)
    is_fake = prob_fake > 0.5
    confidence = prob_fake if is_fake else prob_real

    if is_fake:
        if prob_fake > 0.9:
            risk = "CRITICAL"
        elif prob_fake > 0.75:
            risk = "HIGH"
        else:
            risk = "MEDIUM"
    else:
        risk = "LOW"

    # Generate images
    waveform_img = make_waveform_png(waveform_np, duration)
    spectrogram_img = make_spectrogram_png(waveform_np, 16000, duration)

    # Metrics
    metrics = compute_metrics(waveform_np)
    evidence = build_evidence(metrics)

    return {
        "duration": duration,
        "sample_rate": 16000,
        "is_fake": is_fake,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "confidence": round(confidence, 4),
        "risk": risk,
        "waveform_img": waveform_img,
        "spectrogram_img": spectrogram_img,
        "evidence": evidence,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
