import cv2
import streamlit as st
from model import process_frame
import numpy as np
import tempfile
import time
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import threading
import requests

# ─── APP CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SurakshaNet - AI Surveillance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def render_panel(risk_ph, density_ph, zones_ph, count, crowd_level, crowd_pct, density, zones, behavior, behav_conf):
    """Updates the metrics cards on the right side of the screen."""
    r = crowd_level
    dc = "#10b981" if r=="Low" else ("#f59e0b" if r=="Medium" else "#ef4444")
    bc = "low" if r=="Low" else ("medium" if r=="Medium" else "high")
    dl = "LOW" if density < 0.1 else ("MEDIUM" if density < 0.4 else "HIGH")
    bh = "beh-n" if behavior=="NORMAL" else "beh-s"

    risk_ph.markdown(f"""
    <div class="card">
        <div class="card-title">🛡️ Risk Assessment</div>
        <div style="display:flex;align-items:center;gap:8px;margin:6px 0">
            <span style="width:8px;height:8px;border-radius:50%;background:{dc};display:inline-block;box-shadow:0 0 7px {dc}"></span>
            <span class="brand-name-text" style="color:{dc}; font-size:24px; font-weight:800;">{r.upper()}</span>
        </div>
        <div style="font-size:10px;color:#64748b">Crowd Risk: {crowd_pct}%</div>
        <div style="margin-top:6px"><span class="beh-b {bh}">🧠 {behavior} ({behav_conf:.0f}%)</span></div>
    </div>""", unsafe_allow_html=True)

    density_ph.markdown(f"""
    <div class="card">
        <div class="card-title">🔵 Density</div>
        <span class="rbadge {bc}">{dl}</span>
        <div style="font-size:10px;color:#64748b;margin-top:6px">{density:.3f} index</div>
    </div>""", unsafe_allow_html=True)

    zones_ph.markdown(f"""
    <div class="card">
        <div class="card-title">🗺️ Zones</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px;">
            <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);border-radius:8px;padding:8px 10px;">
                <div style="font-size:9px;color:#64748b;">ZONE A</div><div style="font-size:15px;font-weight:700;color:#10b981;">{zones['A']}</div>
            </div>
            <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);border-radius:8px;padding:8px 10px;">
                <div style="font-size:9px;color:#64748b;">ZONE B</div><div style="font-size:15px;font-weight:700;color:#10b981;">{zones['B']}</div>
            </div>
            <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);border-radius:8px;padding:8px 10px;">
                <div style="font-size:9px;color:#64748b;">ZONE C</div><div style="font-size:15px;font-weight:700;color:#10b981;">{zones['C']}</div>
            </div>
            <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);border-radius:8px;padding:8px 10px;">
                <div style="font-size:9px;color:#64748b;">ZONE D</div><div style="font-size:15px;font-weight:700;color:#10b981;">{zones['D']}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ============================================================
# TELEGRAM ALERT
# ============================================================

def get_telegram_credentials():
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        return token, chat_id
    except Exception:
        return None, None


def send_telegram_alert(risk, count, movement=0, behavior="NORMAL"):
    token, chat_id = get_telegram_credentials()

    if not token or not chat_id:
        return False

    message = (
        "🚨 SURAKSHANET ALERT 🚨\n\n"
        f"Risk Level: {risk}\n"
        f"People Detected: {count}\n"
        f"Movement: {movement:.2f}\n"
        f"Behavior: {behavior}\n\n"
        "Suspicious activity detected."
    )

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": message
            },
            timeout=5
        )

        return response.ok

    except Exception as e:
        print("Telegram error:", e)
        return False


telegram_last_alert = 0
TELEGRAM_COOLDOWN = 15


def maybe_send_alert(risk, count, movement, behavior):
    global telegram_last_alert

    if risk != "High":
        return

    if count <= 10:
        return

    current_time = time.time()

    if current_time - telegram_last_alert < TELEGRAM_COOLDOWN:
        return

    success = send_telegram_alert(
        risk,
        count,
        movement,
        behavior
    )

    if success:
        telegram_last_alert = current_time
        print("🚨 Telegram alert sent")


# ============================================================
# BROWSER WEBCAM PROCESSOR
# ============================================================

class SurakshaNetVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()
        self.last_result = None
        self.frame_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Keep processing resolution controlled
        img = cv2.resize(img, (640, 480))

        try:
            pf, count, dens, cl, cp, zones, beh, bc = process_frame(
                img,
                st.session_state.get("heatmap_on", False),
                st.session_state.get("zones_on", False)
            )

            # Store latest result
            with self.lock:
                self.last_result = {
                    "count": count,
                    "density": dens,
                    "risk": cl,
                    "risk_pct": cp,
                    "zones": zones,
                    "behavior": beh,
                    "behavior_conf": bc
                }

            # Telegram alert
            movement = dens * 100

            maybe_send_alert(
                cl,
                count,
                movement,
                beh
            )

            return av.VideoFrame.from_ndarray(
                pf,
                format="bgr24"
            )

        except Exception as e:

            print("Webcam processing error:", e)

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


# ─── DARK THEME & PREMIUM BRANDING CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:      #0a0d14;
    --bg2:     #111827;
    --card:    #131c2e;
    --card2:   #0f1929;
    --border:  #1e2d45;
    --blue:    #3b82f6;
    --cyan:    #06b6d4;
    --green:   #10b981;
    --red:     #ef4444;
    --yellow:  #f59e0b;
    --txt:     #e2e8f0;
    --txt2:    #64748b;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}
.stApp { background-color: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 1.8rem !important; }

/* 🛡️ BRANDING: LARGE LOGO HIGHLIGHT */
.brand {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 15px;
    padding: 30px 15px;
    background: rgba(59, 130, 246, 0.08);
    border-radius: 20px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    text-align: center;
}
.brand-icon {
    width: 85px; height: 85px;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    border-radius: 22px;
    display: flex; align-items: center; justify-content: center;
    font-size: 45px;
    box-shadow: 0 0 25px rgba(59, 130, 246, 0.6);
}
.brand-name {
    font-weight: 800; font-size: 26px;
    letter-spacing: -0.01em;
    background: linear-gradient(to right, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-sub {
    font-size: 11px; color: var(--cyan);
    font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* CARDS */
.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; margin-bottom: 12px;
}
.card-title { font-size:11px; font-weight:700; text-transform:uppercase; color:var(--cyan); margin-bottom:15px; }

.rbadge { display:inline-flex;padding:5px 14px;border-radius:99px;font-weight:700;font-size:13px; }
.low { background:rgba(16,185,129,.15);color:var(--green); }
.medium { background:rgba(245,158,11,.15);color:var(--yellow); }
.high { background:rgba(239,68,68,.15);color:var(--red); }

.beh-b { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; padding:3px 10px; border-radius:6px; }
.beh-n { background:rgba(16,185,129,.15); color:var(--green); }
.beh-s { background:rgba(239,68,68,.15); color:var(--red); }

.ph { display:flex; align-items:baseline; gap:12px; margin-bottom:20px; padding-bottom:15px; border-bottom:1px solid var(--border); }
.ph h1 { font-size:24px; font-weight:800; color:var(--txt); margin:0; }

[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }

/* TEXT VISIBILITY / READABILITY */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] button p,
section[data-testid="stSidebar"] button span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div {
    color: #e2e8f0 !important;
}

/* SIDEBAR NAVIGATION BUTTONS — FORCE DARK THEME */
section[data-testid="stSidebar"] div.stButton > button,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: #111c31 !important;
    background-color: #111c31 !important;
    color: #f8fafc !important;
    border: 1px solid #263f68 !important;
    border-radius: 16px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    min-height: 58px !important;
    box-shadow: none !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #f8fafc !important;
    transition: all 0.2s ease-in-out !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: #172844 !important;
    background-color: #172844 !important;
    color: #ffffff !important;
    border-color: #2583ff !important;
    box-shadow: 0 0 15px rgba(37, 131, 255, 0.25) !important;
    -webkit-text-fill-color: #ffffff !important;
}

section[data-testid="stSidebar"] div.stButton > button p,
section[data-testid="stSidebar"] div.stButton > button span,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    opacity: 1 !important;
}

div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #e2e8f0 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

div[data-testid="stToggle"] label,
div[data-testid="stToggle"] label p,
div[data-testid="stToggle"] label span {
    color: #e2e8f0 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
.stMarkdown p {
    color: #e2e8f0 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #f1f5f9 !important;
}

/* FORCE SIDEBAR TO REMAIN VISIBLE */
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    min-width: 280px !important;
    width: 280px !important;
    transform: none !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    min-width: 280px !important;
    width: 280px !important;
    transform: none !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.update({
        "page": "Model Architecture",
        "alerts": [],
        "people_count": 0,
        "risk_level": "Low",
        "behavior": "NORMAL",
        "heatmap_on": True,
        "zones_on": True,
        "audio_on": False,
        "sensitivity": 50
    })

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-name">SurakshaNet</div>
            <div class="brand-sub">AI SURVEILLANCE v2.5</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    pages = [("🧠", "Model Architecture"), ("📺", "Live Monitor"), ("📊", "Analytics"), ("🔔", "Alerts")]
    for icon, pg in pages:
        if st.button(f"{icon}  {pg}", use_container_width=True, key=f"nav_{pg}"):
            st.session_state.page = pg
            st.rerun()

page = st.session_state.page

# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Model Architecture":
    st.markdown('<div class="ph"><h1>🧠 Performance & Logic</h1><p>SurakshaNet Backbone v2.0</p></div>', unsafe_allow_html=True)
    if st.button("🚀 Start Surveillance System", type="primary"):
        st.session_state.page = "Live Monitor"
        st.rerun()

    st.markdown(
        "<div style=\"color:#94a3b8;font-size:13px;margin:-8px 0 18px 0;\">"
        "VisDrone person-detection validation results • YOLOv8n • 30 training epochs"
        "</div>",
        unsafe_allow_html=True,
    )

    # Actual results from the completed VisDrone training run.
    metric_cols = st.columns(4)
    metrics = [
        ("Precision", "62.4%"),
        ("Recall", "42.2%"),
        ("mAP@50", "46.4%"),
        ("mAP@50–95", "17.8%"),
    ]
    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            st.markdown(
                f"<div class=\"card\" style=\"text-align:center;min-height:82px;\">"
                f"<div style=\"font-size:10px;color:#64748b;text-transform:uppercase;font-weight:700;\">{label}</div>"
                f"<div style=\"font-size:25px;color:#10b981;font-weight:800;margin-top:7px;\">{value}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    chart_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "mAP@50", "mAP@50–95"],
        "Score": [62.4, 42.2, 46.4, 17.8],
    })
    fig = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        title="VisDrone Validation Metrics",
        text="Score",
        template="plotly_dark",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_yaxes(title="Score (%)", range=[0, 70])
    fig.update_xaxes(title="Detection Metric")
    fig.update_layout(showlegend=False, margin=dict(t=60, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: These are object-detection validation metrics. Traditional classification accuracy is not used as the primary metric for this detector."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Live Monitor":

    st.markdown(
        '<div class="ph"><h1>📺 Real-time Monitoring</h1></div>',
        unsafe_allow_html=True
    )

    col_vid, col_panel = st.columns([3, 1.1])

    # ------------------------------------------------------------
    # RIGHT PANEL
    # ------------------------------------------------------------

    with col_panel:

        risk_ph = st.empty()
        density_ph = st.empty()
        zones_ph = st.empty()

        st.markdown(
            '<div class="card"><div class="card-title">Overlays</div>',
            unsafe_allow_html=True
        )

        st.session_state.heatmap_on = st.toggle(
            "🔥 Heatmap",
            st.session_state.heatmap_on
        )

        st.session_state.zones_on = st.toggle(
            "🗺️ Zones",
            st.session_state.zones_on
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ------------------------------------------------------------
    # VIDEO AREA
    # ------------------------------------------------------------

    with col_vid:

        mode = st.radio(
            "Source",
            [
                "📁 Upload Video",
                "📷 Live Camera"
            ],
            horizontal=True,
            label_visibility="collapsed"
        )

        # ========================================================
        # UPLOAD VIDEO
        # ========================================================

        if "Upload" in mode:

            upl = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mov"],
                key="video_uploader"
            )

            if upl:

                # Save uploaded video temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as tmp:

                    tmp.write(upl.getbuffer())
                    video_path = tmp.name

                st.video(upl)

                st.markdown(
                    "### 🔍 AI Analysis"
                )

                start_analysis = st.button(
                    "▶ Analyze Video",
                    type="primary"
                )

                if start_analysis:

                    cap = cv2.VideoCapture(video_path)

                    if not cap.isOpened():

                        st.error(
                            "Unable to open uploaded video."
                        )

                    else:

                        total_frames = int(
                            cap.get(
                                cv2.CAP_PROP_FRAME_COUNT
                            )
                        )

                        fps = cap.get(
                            cv2.CAP_PROP_FPS
                        )

                        if fps <= 0:
                            fps = 25

                        progress = st.progress(0)

                        frame_placeholder = st.empty()

                        processed = 0
                        last_count = 0

                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break

                            frame = cv2.resize(
                                frame,
                                (640, 480)
                            )

                            try:

                                (
                                    pf,
                                    count,
                                    dens,
                                    cl,
                                    cp,
                                    zones,
                                    beh,
                                    bc
                                ) = process_frame(
                                    frame,
                                    st.session_state.heatmap_on,
                                    st.session_state.zones_on
                                )

                                # Display processed frame
                                frame_placeholder.image(
                                    cv2.cvtColor(
                                        pf,
                                        cv2.COLOR_BGR2RGB
                                    ),
                                    channels="RGB"
                                )

                                render_panel(
                                    risk_ph,
                                    density_ph,
                                    zones_ph,
                                    count,
                                    cl,
                                    cp,
                                    dens,
                                    zones,
                                    beh,
                                    bc
                                )

                                # Telegram alert
                                movement = dens * 100

                                maybe_send_alert(
                                    cl,
                                    count,
                                    movement,
                                    beh
                                )

                                last_count = count

                            except Exception as e:

                                st.warning(
                                    f"Frame processing error: {e}"
                                )

                            processed += 1

                            if total_frames > 0:

                                progress.progress(
                                    min(
                                        processed / total_frames,
                                        1.0
                                    )
                                )

                        cap.release()

                        st.success(
                            f"Analysis complete — "
                            f"{processed} frames processed."
                        )

        # ========================================================
        # BROWSER WEBCAM
        # ========================================================

        else:

            st.markdown(
                "### 📷 Browser Camera"
            )

            st.info(
                "Click START below and allow camera permission "
                "when your browser asks."
            )

            ctx = webrtc_streamer(
                key="surakshanet-camera",

                video_processor_factory=
                    SurakshaNetVideoProcessor,

                media_stream_constraints={
                    "video": True,
                    "audio": False
                },

                async_processing=True
            )

            # ----------------------------------------------------
            # Display latest detection results
            # ----------------------------------------------------

            if ctx.video_processor:

                result = None

                with ctx.video_processor.lock:

                    if ctx.video_processor.last_result:
                        result = dict(
                            ctx.video_processor.last_result
                        )

                if result:

                    render_panel(
                        risk_ph,
                        density_ph,
                        zones_ph,
                        result["count"],
                        result["risk"],
                        result["risk_pct"],
                        result["density"],
                        result["zones"],
                        result["behavior"],
                        result["behavior_conf"]
                    )

                    # Sync analytics
                    st.session_state.people_count = (
                        result["count"]
                    )

                    st.session_state.risk_level = (
                        result["risk"]
                    )

                else:

                    st.caption(
                        "Waiting for camera frames..."
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown('<div class="ph"><h1>📊 System Statistics</h1></div>', unsafe_allow_html=True)
    if os.path.exists("data/logs.csv"):
        df = pd.read_csv("data/logs.csv")
        df["time"] = pd.to_datetime(df["time"], unit="s")
        st.plotly_chart(px.area(df.tail(100), x="time", y="count", title="Crowd Density Trend", template="plotly_dark"), use_container_width=True)
    else:
        st.info("No data available yet.")