import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import io
import time
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CallGuard — AI Voice Fraud Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(96, 165, 250, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.25);
        color: #60a5fa;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Stat cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    /* Result cards */
    .result-card-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.03));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-card-real {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.08), rgba(34, 197, 94, 0.03));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .result-title-fake {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ef4444;
        margin-bottom: 0.3rem;
    }
    .result-title-real {
        font-size: 1.6rem;
        font-weight: 800;
        color: #22c55e;
        margin-bottom: 0.3rem;
    }
    .result-confidence {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-confidence-fake { color: #ef4444; }
    .result-confidence-real { color: #22c55e; }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .risk-critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    .risk-high { background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.4); }
    .risk-medium { background: rgba(234, 179, 8, 0.2); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.4); }
    .risk-low { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.4); }

    /* Info card */
    .info-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .info-card h4 {
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .info-card p, .info-card li {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* Divider styling */
    hr {
        border-color: rgba(148, 163, 184, 0.1) !important;
        margin: 1.5rem 0 !important;
    }

    /* Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ CallGuard</div>
    <div class="hero-subtitle">AI-Powered Voice Fraud Detection</div>
    <div class="hero-badge">Wav2Vec2 Deepfake Detector</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;'>"
    "Upload a call recording to check if the voice is <strong style='color:#e2e8f0;'>real or AI-generated</strong>"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ─── Load Model ─────────────────────────────────────────────────────────────────
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"


@st.cache_resource
def load_model():
    """Load the deepfake detection model and feature extractor."""
    with st.spinner("⏳ Downloading deepfake detection model — first run takes 1-2 min..."):
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        model.eval()
    return model, feature_extractor


model, feature_extractor = load_model()

# ─── File Uploader ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a call audio file",
    type=["wav", "flac", "mp3", "ogg", "m4a"],
    help="Supported formats: WAV, FLAC, MP3, OGG, M4A  •  Best results with 2–13 second clips",
)

if uploaded_file is not None:
    # Play the audio
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)

    # ── Load Audio with librosa ──────────────────────────────────────────────
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        waveform_np, original_sr = librosa.load(audio_buffer, sr=16000, mono=True)
    except Exception as e:
        st.error(f"❌ Could not load audio file: {e}")
        st.stop()

    duration_sec = len(waveform_np) / 16000

    # ── Audio Metadata ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{duration_sec:.1f}s</div>'
            f'<div class="stat-label">Duration</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-value">16 kHz</div>'
            '<div class="stat-label">Sample Rate</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-value">Mono</div>'
            '<div class="stat-label">Channels</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # ── Waveform Visualization ───────────────────────────────────────────────
    st.markdown("### 📊 Voice Waveform")
    fig, ax = plt.subplots(figsize=(10, 2.2))
    time_axis = np.linspace(0, duration_sec, len(waveform_np))
    ax.fill_between(time_axis, waveform_np, alpha=0.3, color="#60a5fa")
    ax.plot(time_axis, waveform_np, color="#60a5fa", linewidth=0.4)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Amplitude", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.set_xlim(0, duration_sec)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Mel Spectrogram ──────────────────────────────────────────────────────
    st.markdown("### 🔬 Voice Frequency Analysis")
    mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=16000, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    img = ax2.imshow(
        mel_spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, duration_sec, 0, 64],
    )
    ax2.set_facecolor("#0f172a")
    fig2.patch.set_facecolor("#0f172a")
    ax2.set_xlabel("Time (s)", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Mel Bin", color="#94a3b8", fontsize=9)
    ax2.tick_params(colors="#64748b", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#1e293b")
    cbar = fig2.colorbar(img, ax=ax2, format="%+.0f dB", pad=0.02)
    cbar.ax.tick_params(colors="#64748b", labelsize=8)
    cbar.outline.set_edgecolor("#1e293b")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Analyze Button ───────────────────────────────────────────────────────
    st.markdown("")
    if st.button("🔍 Analyze Voice", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Preparing audio features...")

        # Step 1: Extract features
        inputs = feature_extractor(
            waveform_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        progress_bar.progress(40, text="Running deepfake detection model...")

        # Step 2: Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        progress_bar.progress(80, text="Processing results...")

        # Class 0 = Real, Class 1 = Fake (per model card)
        prob_real = probs[0][0].item()
        prob_fake = probs[0][1].item()
        is_fake = prob_fake > 0.5
        confidence = prob_fake if is_fake else prob_real

        # Risk level
        if is_fake:
            if prob_fake > 0.9:
                risk_level, risk_class = "CRITICAL", "risk-critical"
            elif prob_fake > 0.75:
                risk_level, risk_class = "HIGH", "risk-high"
            else:
                risk_level, risk_class = "MEDIUM", "risk-medium"
        else:
            risk_level, risk_class = "LOW", "risk-low"

        progress_bar.progress(100, text="Analysis complete!")
        time.sleep(0.3)
        progress_bar.empty()

        st.divider()

        # ── Result Card ──────────────────────────────────────────────────────
        if is_fake:
            st.markdown(
                f"""
                <div class="result-card-fake">
                    <div class="result-icon">🚨</div>
                    <div class="result-title-fake">AI-GENERATED VOICE DETECTED</div>
                    <div class="result-confidence result-confidence-fake">{confidence * 100:.1f}%</div>
                    <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.8rem;">Fake Probability</div>
                    <span class="{risk_class} risk-badge">⚠ {risk_level} RISK</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card-real">
                    <div class="result-icon">✅</div>
                    <div class="result-title-real">GENUINE HUMAN VOICE</div>
                    <div class="result-confidence result-confidence-real">{confidence * 100:.1f}%</div>
                    <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.8rem;">Real Probability</div>
                    <span class="{risk_class} risk-badge">✓ {risk_level} RISK</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Probability Breakdown ────────────────────────────────────────────
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value" style="color:#22c55e;">{prob_real * 100:.1f}%</div>'
                f'<div class="stat-label">Real Probability</div></div>',
                unsafe_allow_html=True,
            )
        with prob_col2:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value" style="color:#ef4444;">{prob_fake * 100:.1f}%</div>'
                f'<div class="stat-label">Fake Probability</div></div>',
                unsafe_allow_html=True,
            )

        # ── Confidence Gauge ─────────────────────────────────────────────────
        st.markdown("### 📈 Confidence Gauge")
        fig3, ax3 = plt.subplots(figsize=(8, 1.2))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        # Background gradient bar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax3.imshow(
            gradient,
            aspect="auto",
            cmap="RdYlGn_r",
            extent=[0, 1, 0.25, 0.75],
            alpha=0.7,
        )

        # Marker for fake probability
        marker_x = prob_fake
        ax3.plot(marker_x, 0.5, "v", color="white", markersize=14, zorder=5)
        ax3.plot(marker_x, 0.5, "v", color="#0f172a", markersize=10, zorder=6)

        # Labels
        ax3.text(0, 0.1, "REAL", color="#22c55e", fontsize=9, fontweight="bold", ha="left")
        ax3.text(1, 0.1, "FAKE", color="#ef4444", fontsize=9, fontweight="bold", ha="right")
        ax3.text(0.5, 0.1, "UNCERTAIN", color="#eab308", fontsize=8, fontweight="bold", ha="center")

        ax3.set_facecolor("#0f172a")
        fig3.patch.set_facecolor("#0f172a")
        ax3.axis("off")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # ── Advice ───────────────────────────────────────────────────────────
        if is_fake:
            st.markdown(
                """
                <div class="info-card">
                    <h4>⚠️ What You Should Do</h4>
                    <ul>
                        <li>Do <strong>NOT</strong> share OTPs, PINs, or financial information</li>
                        <li>Hang up and call the person/organization directly on their official number</li>
                        <li>Report the call to your bank or local cyber-crime helpline</li>
                        <li>Save the recording as evidence if possible</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="info-card">
                    <h4>✅ Voice Appears Genuine</h4>
                    <p>The voice exhibits natural speech patterns consistent with a real human speaker.
                    However, always exercise caution with unexpected calls asking for sensitive information.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Model Info ───────────────────────────────────────────────────────
        with st.expander("ℹ️ About This Analysis"):
            st.markdown(f"""
            - **Model**: `{MODEL_NAME}`
            - **Architecture**: Wav2Vec2 fine-tuned for binary deepfake detection
            - **Classes**: Real (Class 0) vs Fake (Class 1)
            - **Training Data**: ElevenLabs, Amazon Polly, Kokoro, Hume AI, Speechify, Luvvoice
            - **Validation Accuracy**: 97.9% on held-out test set
            - **Audio Processed**: {duration_sec:.1f}s at 16 kHz mono

            > ⚠️ **Disclaimer**: CallGuard is an AI tool designed to assist in detecting
            > AI-generated voices. It may not catch all deepfakes, especially from newer
            > TTS systems not in the training data. Always verify suspicious calls
            > through official channels.
            """)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.75rem;'>"
    "CallGuard v1.0 — Built with Wav2Vec2 & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)