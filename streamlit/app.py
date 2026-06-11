import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR Screening | Retinal AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Medical / Clinical Design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp {
    background: #f0f4f8;
    color: #1a202c;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0; }

/* ── Top header bar ── */
.top-bar {
    background: linear-gradient(90deg, #1a56db 0%, #1e429f 100%);
    padding: 14px 32px;
    display: flex;
    align-items: center;
    gap: 14px;
    border-radius: 14px;
    margin-bottom: 22px;
    user-select: none;
}
.top-bar-icon {
    width: 44px; height: 44px;
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
}
.top-bar-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.3px;
}
.top-bar-sub {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75);
    margin: 2px 0 0 0;
}
.top-bar-badge {
    margin-left: auto;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 20px;
    white-space: nowrap;
}

/* ── White cards ── */
.med-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px 26px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #718096;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
    user-select: none;
}
.card-heading {
    font-size: 1rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
}

/* ── Severity result block ── */
.severity-block {
    border-left: 5px solid var(--sev-color);
    border-radius: 0 12px 12px 0;
    background: linear-gradient(90deg, color-mix(in srgb, var(--sev-color) 8%, white), #ffffff);
    padding: 18px 22px;
    margin-bottom: 16px;
}
.severity-grade {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--sev-color);
    margin-bottom: 4px;
    user-select: none;
}
.severity-name {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1a202c;
    line-height: 1.1;
    margin-bottom: 4px;
    user-select: none;
}
.severity-conf {
    font-size: 0.85rem;
    color: #718096;
    user-select: none;
}

/* ── Stage indicators ── */
.stage-row {
    display: flex;
    gap: 8px;
    margin: 14px 0 6px;
}
.stage-pill {
    flex: 1;
    text-align: center;
    padding: 7px 4px;
    border-radius: 8px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    border: 2px solid transparent;
    color: #718096;
    background: #f7fafc;
    transition: all 0.2s;
    user-select: none;
}
.stage-pill.active {
    color: #fff;
    border-color: transparent;
}

/* ── Info row ── */
.info-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 16px;
}
.info-cell {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
}
.info-cell-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #718096;
    margin-bottom: 6px;
    user-select: none;
}
.info-cell-value {
    font-size: 0.88rem;
    color: #2d3748;
    line-height: 1.5;
    font-weight: 400;
}

/* ── Urgency banner ── */
.urgency-banner {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 14px;
    border-left: 4px solid var(--sev-color);
    background: linear-gradient(90deg, color-mix(in srgb, var(--sev-color) 6%, white), #f7fafc);
}
.urgency-icon { font-size: 1.4rem; flex-shrink: 0; line-height: 1; }
.urgency-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #718096;
    margin-bottom: 3px;
    user-select: none;
}
.urgency-text {
    font-size: 0.88rem;
    color: #2d3748;
    line-height: 1.5;
}

/* ── Idle placeholder ── */
.idle-box {
    background: #ffffff;
    border: 2px dashed #cbd5e0;
    border-radius: 14px;
    padding: 56px 24px;
    text-align: center;
}
.idle-icon { font-size: 3rem; }
.idle-text { margin-top: 12px; font-size: 0.95rem; color: #a0aec0; }

/* ── Disclaimer ── */
.disclaimer-bar {
    background: #fffbeb;
    border: 1px solid #f6e05e;
    border-left: 4px solid #d69e2e;
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.82rem;
    color: #744210;
    margin-top: 18px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    line-height: 1.5;
}
.disclaimer-bar strong { color: #92400e; }

/* ── Streamlit overrides ── */

/* File uploader — full white medical styling */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #f7fafc !important;
    border: 2px dashed #90cdf4 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: #ebf8ff !important;
    border-color: #1a56db !important;
}
[data-testid="stFileUploadDropzone"] > div {
    color: #4a5568 !important;
}
[data-testid="stFileUploadDropzone"] span {
    color: #4a5568 !important;
    font-size: 0.9rem !important;
}
[data-testid="stFileUploadDropzone"] small {
    color: #718096 !important;
}
/* Upload cloud icon */
[data-testid="stFileUploadDropzone"] svg {
    fill: #1a56db !important;
    color: #1a56db !important;
}
/* Browse files button inside uploader */
[data-testid="stFileUploadDropzone"] button,
[data-testid="baseButton-secondary"] {
    background: #1a56db !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    box-shadow: 0 2px 6px rgba(26,86,219,0.25) !important;
}
[data-testid="stFileUploadDropzone"] button:hover {
    background: #1e429f !important;
}

/* Analyze button */
.stButton > button {
    background: linear-gradient(135deg, #1a56db, #1e429f) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 12px 28px !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 8px rgba(26,86,219,0.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 18px rgba(26,86,219,0.45) !important;
    transform: translateY(-1px) !important;
}
.stSpinner > div { border-top-color: #1a56db !important; }


/* ── Sidebar content ── */
.sb-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #a0aec0;
    margin: 20px 0 10px;
    padding: 0 20px;
    user-select: none;
}
.sb-divider { height: 1px; background: #e2e8f0; margin: 4px 0; }
.sb-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 20px;
    font-size: 0.84rem;
}
.sb-info-row span:first-child { color: #718096; }
.sb-info-row span:last-child  { color: #2d3748; font-weight: 500; }
.sb-scale-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 20px;
}
.sb-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sb-scale-name { font-size: 0.85rem; color: #4a5568; }
.sb-scale-stage { font-size: 0.72rem; color: #a0aec0; margin-left: auto; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Clinical Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
CLASS_STAGES  = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
CLASS_COLORS  = ["#16a34a", "#d97706", "#ea580c", "#dc2626", "#9333ea"]
CLASS_BG      = ["#dcfce7", "#fef3c7", "#ffedd5", "#fee2e2", "#f3e8ff"]

CLASS_DESCRIPTIONS = [
    "No signs of diabetic retinopathy. Retinal vessels appear normal. Continue routine annual screening.",
    "Microaneurysms only — earliest detectable stage. Small bulges in blood vessel walls noted.",
    "Moderate non-proliferative DR. More than microaneurysms, less than severe. Dot hemorrhages and hard exudates present.",
    "Severe NPDR. Extensive intraretinal hemorrhages in all four quadrants, venous beading, or IRMA observed.",
    "Proliferative DR. Neovascularization or vitreous/pre-retinal hemorrhage present. Vision-threatening stage.",
]

URGENCY_DATA = [
    ("🟢", "Routine Follow-up",   "No immediate intervention required. Schedule next eye exam in 12 months."),
    ("🟡", "Early Monitoring",    "Recommend ophthalmologist follow-up within 6–12 months. Optimize glycemic control."),
    ("🟠", "Timely Referral",     "Refer to ophthalmologist within 3–6 months. Consider fluorescein angiography."),
    ("🔴", "Urgent Referral",     "Refer to retinal specialist within 1 month. Risk of vision loss is elevated."),
    ("🚨", "Immediate Referral",  "Urgent consultation with vitreoretinal surgeon required. High risk of blindness."),
]

STAGE_PILL_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Prolif."]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_convnext_model.pth")
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Model & Transform
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image: Image.Image, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return int(np.argmax(probs)), probs

# ─────────────────────────────────────────────────────────────────────────────
# Plotly Chart
# ─────────────────────────────────────────────────────────────────────────────
def make_chart(probs, pred_idx):
    colors = [CLASS_COLORS[i] if i == pred_idx else "#e2e8f0" for i in range(5)]
    fig = go.Figure(go.Bar(
        x=STAGE_PILL_LABELS,
        y=(probs * 100).tolist(),
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#4a5568", size=11, family="Inter"),
        hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#718096", size=11),
        margin=dict(t=28, b=8, l=0, r=0),
        yaxis=dict(
            range=[0, 115],
            showgrid=True,
            gridcolor="#f0f4f8",
            ticksuffix="%",
            tickfont=dict(size=10, color="#a0aec0"),
            zeroline=False,
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#4a5568"),
        ),
        showlegend=False,
        height=260,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a56db,#1e429f);padding:24px 20px 20px;margin-bottom:4px;">
        <div style="font-size:1.1rem;font-weight:700;color:#fff;user-select:none;">🏥 Retinal AI</div>
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.7);margin-top:3px;user-select:none;">DR Screening System v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">About This Tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="padding:0 20px 12px;font-size:0.84rem;color:#4a5568;line-height:1.6;">'
        'An AI-powered screening tool using <strong>ConvNeXt-Tiny</strong> to classify '
        'Diabetic Retinopathy severity from retinal fundus photographs across the '
        '<strong>ICDR 5-class scale</strong>.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-title">DR Severity Scale</div>', unsafe_allow_html=True)

    scale_data = [
        ("No DR",           "Stage 0", CLASS_COLORS[0]),
        ("Mild NPDR",       "Stage 1", CLASS_COLORS[1]),
        ("Moderate NPDR",   "Stage 2", CLASS_COLORS[2]),
        ("Severe NPDR",     "Stage 3", CLASS_COLORS[3]),
        ("Proliferative",   "Stage 4", CLASS_COLORS[4]),
    ]
    for name, stage, color in scale_data:
        st.markdown(
            f'<div class="sb-scale-row">'
            f'<div class="sb-dot" style="background:{color};"></div>'
            f'<span class="sb-scale-name">{name}</span>'
            f'<span class="sb-scale-stage">{stage}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sb-divider" style="margin-top:8px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-title">Model Information</div>', unsafe_allow_html=True)

    model_rows = [
        ("Architecture", "ConvNeXt-Tiny"),
        ("Input Size",   "224 × 224 px"),
        ("Classes",      "5 (ICDR Scale)"),
        ("Device",       str(DEVICE).upper()),
    ]
    for label, val in model_rows:
        st.markdown(
            f'<div class="sb-info-row"><span>{label}</span><span>{val}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sb-divider" style="margin-top:8px;"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="padding:12px 20px;font-size:0.75rem;color:#a0aec0;line-height:1.5;">'
        '⚕️ For clinical decision support only. Not a substitute for professional diagnosis.'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Top Bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="top-bar-icon">🔬</div>
  <div>
    <div class="top-bar-title">Diabetic Retinopathy Screening</div>
    <div class="top-bar-sub">Upload a retinal fundus image for automated DR severity grading</div>
  </div>
  <div class="top-bar-badge">AI · ConvNeXt-Tiny</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.15], gap="large")

# ── LEFT: Upload ──────────────────────────────────────────────────────────────
with col_left:
    # Upload card
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">📁 &nbsp;Fundus Image Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Retinal fundus photograph",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a retinal fundus photograph (JPG, PNG, BMP, TIFF supported)",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded fundus image", use_column_width="always")
        st.markdown("</div>", unsafe_allow_html=True)  # close med-card

        # Image info card
        w, h = image.size
        st.markdown(f"""
        <div class="med-card" style="padding:16px 22px;">
          <div class="card-label">Image Details</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px;">
            <div style="text-align:center;">
              <div style="font-size:0.7rem;color:#718096;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Width</div>
              <div style="font-weight:600;color:#2d3748;">{w} px</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:0.7rem;color:#718096;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Height</div>
              <div style="font-weight:600;color:#2d3748;">{h} px</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:0.7rem;color:#718096;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Mode</div>
              <div style="font-weight:600;color:#2d3748;">{image.mode}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.button("🔍  Run DR Analysis", key="analyze_btn")
    else:
        st.markdown(
            '<div style="border:2px dashed #cbd5e0;border-radius:10px;padding:36px 16px;text-align:center;margin-top:4px;">'
            '<div style="font-size:2.5rem;">🏥</div>'
            '<div style="margin-top:10px;font-size:0.88rem;color:#a0aec0;">Drag & drop or click above to upload<br>a retinal fundus photograph</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)  # close med-card

# ── RIGHT: Results ────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card-heading" style="margin-bottom:14px;">📋 &nbsp;Clinical Report</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div class="idle-box">
          <div class="idle-icon">🔍</div>
          <div class="idle-text">Upload a fundus image to generate<br>the automated DR screening report</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Run inference
        run_btn = st.session_state.get("analyze_btn", False)
        if run_btn:
            with st.spinner("Analyzing retinal image…"):
                try:
                    model = load_model()
                    pred_idx, probs = predict(image, model)
                    st.session_state["pred_idx"] = pred_idx
                    st.session_state["probs"]    = probs
                except Exception as e:
                    st.error(f"❌ Inference error: {e}")
                    st.stop()

        if "pred_idx" in st.session_state:
            pred_idx   = st.session_state["pred_idx"]
            probs      = st.session_state["probs"]
            color      = CLASS_COLORS[pred_idx]
            confidence = probs[pred_idx] * 100
            urg_icon, urg_title, urg_msg = URGENCY_DATA[pred_idx]

            # ── Severity result ──
            st.markdown(f"""
            <div class="severity-block" style="--sev-color:{color};">
              <div class="severity-grade">Predicted Diagnosis</div>
              <div class="severity-name">{CLASS_NAMES[pred_idx]}</div>
              <div class="severity-conf">Confidence Score: <strong>{confidence:.1f}%</strong> &nbsp;·&nbsp; {CLASS_STAGES[pred_idx]}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Stage pill indicator ──
            pills = ""
            for i, label in enumerate(STAGE_PILL_LABELS):
                if i == pred_idx:
                    pills += (f'<div class="stage-pill active" '
                              f'style="background:{CLASS_COLORS[i]};">{label}</div>')
                else:
                    pills += f'<div class="stage-pill">{label}</div>'
            st.markdown(
                f'<div class="card-label" style="margin-top:4px;">Severity Progression</div>'
                f'<div class="stage-row">{pills}</div>',
                unsafe_allow_html=True
            )

            # ── Probability chart ──
            st.markdown('<div class="card-label" style="margin-top:16px;">Probability Distribution</div>', unsafe_allow_html=True)
            fig = make_chart(probs, pred_idx)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # ── Clinical details ──
            st.markdown(f"""
            <div class="info-row">
              <div class="info-cell">
                <div class="info-cell-label">🔬 Clinical Findings</div>
                <div class="info-cell-value">{CLASS_DESCRIPTIONS[pred_idx]}</div>
              </div>
              <div class="info-cell">
                <div class="info-cell-label">📅 Management Plan</div>
                <div class="info-cell-value">{urg_msg}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Urgency banner ──
            st.markdown(f"""
            <div class="urgency-banner" style="--sev-color:{color};">
              <div class="urgency-icon">{urg_icon}</div>
              <div>
                <div class="urgency-title">{urg_title}</div>
                <div class="urgency-text">{urg_msg}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="idle-box" style="border-style:solid;border-color:#bee3f8;background:#ebf8ff;">
              <div class="idle-icon">⬅️</div>
              <div class="idle-text" style="color:#2b6cb0;">Click <strong>Run DR Analysis</strong><br>to generate the clinical report</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Medical Disclaimer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-bar">
  <span style="font-size:1.1rem;">⚠️</span>
  <div>
    <strong>Medical Disclaimer:</strong> This tool is intended for research and educational screening assistance only.
    Results must be interpreted by a qualified ophthalmologist. This system is
    <strong>not</strong> a substitute for professional medical examination, diagnosis, or treatment.
    Always consult a licensed eye-care professional for clinical decisions.
  </div>
</div>
""", unsafe_allow_html=True)
