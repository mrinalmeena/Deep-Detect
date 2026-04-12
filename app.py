import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import io
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from audio_recorder_streamlit import audio_recorder

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepDetect",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Tighten spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1100px;
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.3rem;
    }
    .app-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.5px;
    }
    .app-desc {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }

    /* Mode selector tabs */
    .mode-tabs {
        display: flex;
        gap: 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #1e293b;
    }
    .mode-tab {
        padding: 0.6rem 1.5rem;
        font-size: 0.88rem;
        font-weight: 500;
        color: #64748b;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        transition: all 0.2s;
    }
    .mode-tab-active {
        color: #e2e8f0;
        border-bottom: 2px solid #3b82f6;
    }

    /* Compact stat */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin: 0.8rem 0;
    }
    .stat-pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.45rem 0.9rem;
        font-size: 0.78rem;
        color: #94a3b8;
    }
    .stat-pill strong {
        color: #e2e8f0;
        margin-right: 0.3rem;
    }

    /* Result strip */
    .result-strip {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.4rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-strip-fake {
        background: #1c1114;
        border: 1px solid #5c2230;
    }
    .result-strip-real {
        background: #0f1d15;
        border: 1px solid #1a4d2e;
    }
    .result-label {
        font-size: 1.15rem;
        font-weight: 700;
    }
    .result-label-fake { color: #f87171; }
    .result-label-real { color: #4ade80; }
    .result-conf {
        font-size: 1.6rem;
        font-weight: 700;
        margin-left: auto;
    }
    .result-conf-fake { color: #f87171; }
    .result-conf-real { color: #4ade80; }
    .risk-tag {
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tag-critical { background: #3b1119; color: #f87171; }
    .tag-high { background: #3b2010; color: #fb923c; }
    .tag-medium { background: #3b3510; color: #facc15; }
    .tag-low { background: #0f3b1d; color: #4ade80; }

    /* Metric evidence table */
    .evidence-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.8rem 0;
        font-size: 0.82rem;
    }
    .evidence-table th {
        text-align: left;
        padding: 0.5rem 0.7rem;
        color: #64748b;
        font-weight: 600;
        border-bottom: 1px solid #1e293b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .evidence-table td {
        padding: 0.5rem 0.7rem;
        color: #cbd5e1;
        border-bottom: 1px solid #1e293b;
    }
    .indicator-bar {
        height: 6px;
        border-radius: 3px;
        background: #1e293b;
        position: relative;
        min-width: 100px;
    }
    .indicator-fill {
        height: 100%;
        border-radius: 3px;
        position: absolute;
        top: 0; left: 0;
    }
    .fill-suspect { background: #f87171; }
    .fill-normal { background: #4ade80; }
    .fill-neutral { background: #fbbf24; }

    /* Prob bars */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
    }
    .prob-label {
        width: 40px;
        color: #94a3b8;
        font-weight: 500;
    }
    .prob-bar-bg {
        flex: 1;
        height: 8px;
        background: #1e293b;
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-bar-fill-real {
        height: 100%;
        background: #4ade80;
        border-radius: 4px;
    }
    .prob-bar-fill-fake {
        height: 100%;
        background: #f87171;
        border-radius: 4px;
    }
    .prob-val {
        width: 50px;
        text-align: right;
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Recording indicator */
    .rec-prompt {
        text-align: center;
        color: #94a3b8;
        padding: 2rem 1rem;
        font-size: 0.9rem;
    }
    .rec-prompt strong {
        color: #e2e8f0;
    }

    /* Section label */
    .section-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #475569;
        font-weight: 600;
        margin-bottom: 0.4rem;
        margin-top: 1rem;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span class="app-title">🔍 DeepDetect</span>
</div>
<div class="app-desc">Audio deepfake detection — upload a file or record live from your microphone.</div>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────────
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"


@st.cache_resource
def load_model():
    """Load the deepfake detection model and feature extractor."""
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return model, feature_extractor


with st.spinner("Loading model..."):
    model, feature_extractor = load_model()


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def compute_audio_metrics(waveform_np, sr=16000):
    """Compute detailed audio metrics for deepfake evidence analysis."""
    metrics = {}

    # 1. Pitch variability (real voices have more natural variation)
    pitches, magnitudes = librosa.piptrack(y=waveform_np, sr=sr, fmin=50, fmax=400)
    pitch_values = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        if p > 0:
            pitch_values.append(p)
    if len(pitch_values) > 2:
        pitch_arr = np.array(pitch_values)
        metrics["pitch_std"] = float(np.std(pitch_arr))
        metrics["pitch_mean"] = float(np.mean(pitch_arr))
        metrics["pitch_variability"] = float(np.std(pitch_arr) / (np.mean(pitch_arr) + 1e-8))
    else:
        metrics["pitch_std"] = 0.0
        metrics["pitch_mean"] = 0.0
        metrics["pitch_variability"] = 0.0

    # 2. Spectral flatness (closer to 1 = more noise-like / synthetic)
    spec_flat = librosa.feature.spectral_flatness(y=waveform_np)
    metrics["spectral_flatness"] = float(np.mean(spec_flat))

    # 3. Zero crossing rate variability
    zcr = librosa.feature.zero_crossing_rate(waveform_np)
    metrics["zcr_mean"] = float(np.mean(zcr))
    metrics["zcr_std"] = float(np.std(zcr))

    # 4. RMS energy variability (real voices have dynamic range)
    rms = librosa.feature.rms(y=waveform_np)
    metrics["rms_variability"] = float(np.std(rms) / (np.mean(rms) + 1e-8))

    # 5. Spectral rolloff consistency
    rolloff = librosa.feature.spectral_rolloff(y=waveform_np, sr=sr)
    metrics["rolloff_std"] = float(np.std(rolloff))

    return metrics


def compute_evidence_rows(metrics, is_fake):
    """Return evidence rows: (metric_name, value_display, assessment, bar_pct, bar_class)."""
    rows = []

    # Pitch variability
    pv = metrics["pitch_variability"]
    if pv < 0.08:
        assessment, cls = "Unnaturally steady", "fill-suspect"
    elif pv > 0.25:
        assessment, cls = "Natural variation", "fill-normal"
    else:
        assessment, cls = "Moderate", "fill-neutral"
    rows.append(("Pitch Variability", f"{pv:.3f}", assessment, min(pv / 0.3 * 100, 100), cls))

    # Spectral flatness
    sf = metrics["spectral_flatness"]
    if sf > 0.05:
        assessment, cls = "Synthetic texture", "fill-suspect"
    elif sf < 0.01:
        assessment, cls = "Natural spectrum", "fill-normal"
    else:
        assessment, cls = "Borderline", "fill-neutral"
    rows.append(("Spectral Flatness", f"{sf:.4f}", assessment, min(sf / 0.08 * 100, 100), cls))

    # RMS energy variability
    rv = metrics["rms_variability"]
    if rv < 0.3:
        assessment, cls = "Too uniform", "fill-suspect"
    elif rv > 0.6:
        assessment, cls = "Natural dynamics", "fill-normal"
    else:
        assessment, cls = "Moderate", "fill-neutral"
    rows.append(("Energy Dynamics", f"{rv:.3f}", assessment, min(rv / 1.0 * 100, 100), cls))

    # Zero crossing rate
    zcr = metrics["zcr_std"]
    if zcr < 0.01:
        assessment, cls = "Low variation", "fill-suspect"
    else:
        assessment, cls = "Normal variation", "fill-normal"
    rows.append(("ZCR Variation", f"{zcr:.4f}", assessment, min(zcr / 0.05 * 100, 100), cls))

    # Spectral rolloff
    ro = metrics["rolloff_std"]
    if ro < 200:
        assessment, cls = "Rigid spectrum", "fill-suspect"
    else:
        assessment, cls = "Dynamic spectrum", "fill-normal"
    rows.append(("Spectral Rolloff Std", f"{ro:.0f} Hz", assessment, min(ro / 1000 * 100, 100), cls))

    return rows


def run_analysis(waveform_np, sr=16000):
    """Run the full analysis pipeline and render results."""

    duration_sec = len(waveform_np) / sr

    # ── Audio stats ──────────────────────────────────────────────────────
    st.markdown(
        f'<div class="stat-row">'
        f'<span class="stat-pill"><strong>{duration_sec:.1f}s</strong> duration</span>'
        f'<span class="stat-pill"><strong>16 kHz</strong> sample rate</span>'
        f'<span class="stat-pill"><strong>mono</strong> channel</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Graphs side by side ──────────────────────────────────────────────
    st.markdown('<div class="section-label">Signal Analysis</div>', unsafe_allow_html=True)
    col_wave, col_spec = st.columns(2)

    with col_wave:
        st.markdown("**Waveform — Amplitude over Time**")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        time_axis = np.linspace(0, duration_sec, len(waveform_np))
        ax1.fill_between(time_axis, waveform_np, alpha=0.25, color="#60a5fa")
        ax1.plot(time_axis, waveform_np, color="#60a5fa", linewidth=0.4)
        ax1.set_facecolor("#0f172a")
        fig1.patch.set_facecolor("#0f172a")
        ax1.set_xlabel("Time (s)", color="#64748b", fontsize=9)
        ax1.set_ylabel("Amplitude", color="#64748b", fontsize=9)
        ax1.tick_params(colors="#475569", labelsize=8)
        for s in ax1.spines.values():
            s.set_color("#1e293b")
        ax1.set_xlim(0, duration_sec)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_spec:
        st.markdown("**Mel Spectrogram — Frequency over Time**")
        mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=sr, n_mels=80)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        img = ax2.imshow(mel_db, aspect="auto", origin="lower", cmap="inferno",
                         extent=[0, duration_sec, 0, 80])
        ax2.set_facecolor("#0f172a")
        fig2.patch.set_facecolor("#0f172a")
        ax2.set_xlabel("Time (s)", color="#64748b", fontsize=9)
        ax2.set_ylabel("Mel Bin", color="#64748b", fontsize=9)
        ax2.tick_params(colors="#475569", labelsize=8)
        for s in ax2.spines.values():
            s.set_color("#1e293b")
        cbar = fig2.colorbar(img, ax=ax2, format="%+.0f dB", pad=0.02)
        cbar.ax.tick_params(colors="#475569", labelsize=8)
        cbar.outline.set_edgecolor("#1e293b")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Model Prediction ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Detection Result</div>', unsafe_allow_html=True)

    inputs = feature_extractor(
        waveform_np, sampling_rate=16000, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    prob_real = probs[0][0].item()
    prob_fake = probs[0][1].item()
    is_fake = prob_fake > 0.5
    confidence = prob_fake if is_fake else prob_real

    # Risk level
    if is_fake:
        if prob_fake > 0.9:
            risk, tag_cls = "CRITICAL", "tag-critical"
        elif prob_fake > 0.75:
            risk, tag_cls = "HIGH", "tag-high"
        else:
            risk, tag_cls = "MEDIUM", "tag-medium"
    else:
        risk, tag_cls = "LOW", "tag-low"

    # Result strip
    if is_fake:
        st.markdown(
            f'<div class="result-strip result-strip-fake">'
            f'<span class="result-label result-label-fake">🚨 AI-Generated Voice Detected</span>'
            f'<span class="risk-tag {tag_cls}">{risk}</span>'
            f'<span class="result-conf result-conf-fake">{confidence * 100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-strip result-strip-real">'
            f'<span class="result-label result-label-real">✓ Genuine Human Voice</span>'
            f'<span class="risk-tag {tag_cls}">{risk}</span>'
            f'<span class="result-conf result-conf-real">{confidence * 100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Probability bars
    st.markdown(
        f'<div class="prob-row">'
        f'<span class="prob-label">Real</span>'
        f'<div class="prob-bar-bg"><div class="prob-bar-fill-real" style="width:{prob_real*100:.1f}%"></div></div>'
        f'<span class="prob-val">{prob_real*100:.1f}%</span>'
        f'</div>'
        f'<div class="prob-row">'
        f'<span class="prob-label">Fake</span>'
        f'<div class="prob-bar-bg"><div class="prob-bar-fill-fake" style="width:{prob_fake*100:.1f}%"></div></div>'
        f'<span class="prob-val">{prob_fake*100:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Evidence / Metric Analysis ───────────────────────────────────────
    st.markdown('<div class="section-label">What indicates deepfake?</div>', unsafe_allow_html=True)

    metrics = compute_audio_metrics(waveform_np, sr)
    evidence_rows = compute_evidence_rows(metrics, is_fake)

    table_html = '<table class="evidence-table"><thead><tr>'
    table_html += '<th>Metric</th><th>Value</th><th>Assessment</th><th>Indicator</th>'
    table_html += '</tr></thead><tbody>'
    for name, val, assessment, pct, cls in evidence_rows:
        table_html += (
            f'<tr><td>{name}</td><td>{val}</td><td>{assessment}</td>'
            f'<td><div class="indicator-bar"><div class="indicator-fill {cls}" '
            f'style="width:{pct:.0f}%"></div></div></td></tr>'
        )
    table_html += '</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)

    # Brief explanation
    if is_fake:
        suspects = [name for name, _, _, _, cls in evidence_rows if cls == "fill-suspect"]
        if suspects:
            st.markdown(
                f"<p style='color:#94a3b8; font-size:0.82rem; margin-top:0.5rem;'>"
                f"<strong style='color:#f87171;'>Key signals:</strong> "
                f"{', '.join(suspects)} — these metrics show patterns typical of AI-synthesized audio.</p>",
                unsafe_allow_html=True,
            )
    else:
        normals = [name for name, _, _, _, cls in evidence_rows if cls == "fill-normal"]
        if normals:
            st.markdown(
                f"<p style='color:#94a3b8; font-size:0.82rem; margin-top:0.5rem;'>"
                f"<strong style='color:#4ade80;'>Natural signals:</strong> "
                f"{', '.join(normals)} — these metrics show patterns consistent with human speech.</p>",
                unsafe_allow_html=True,
            )


# ─── Mode Selection ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Choose input mode",
    ["📁 Upload Audio File", "🎙️ Record Live"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("")

# ─── Mode 1: Upload ─────────────────────────────────────────────────────────────
if mode == "📁 Upload Audio File":
    uploaded_file = st.file_uploader(
        "Drop an audio file here",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes)

        try:
            waveform_np, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            st.error(f"Could not read audio: {e}")
            st.stop()

        run_analysis(waveform_np)

# ─── Mode 2: Live Record ────────────────────────────────────────────────────────
elif mode == "🎙️ Record Live":
    st.markdown(
        '<div class="rec-prompt">'
        'Click the microphone below to <strong>start recording</strong>. '
        'Click again to <strong>stop</strong>. The analysis runs automatically.'
        '</div>',
        unsafe_allow_html=True,
    )

    audio_data = audio_recorder(
        text="",
        recording_color="#f87171",
        neutral_color="#64748b",
        icon_size="2x",
        pause_threshold=60.0,
        sample_rate=16000,
    )

    if audio_data is not None:
        st.audio(audio_data, format="audio/wav")

        try:
            waveform_np, _ = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
        except Exception as e:
            st.error(f"Could not process recording: {e}")
            st.stop()

        if len(waveform_np) / 16000 < 0.5:
            st.warning("Recording too short — speak for at least 1 second.")
        else:
            run_analysis(waveform_np)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown(
    "<p style='text-align:center; color:#334155; font-size:0.7rem; margin-top:2rem;'>"
    "DeepDetect v1.0 — Wav2Vec2 Audio Classification"
    "</p>",
    unsafe_allow_html=True,
)