import streamlit as st
import cv2
import time
from model import predict_risk

# Page config
st.set_page_config(page_title="SurakshaNet", layout="wide")

st.title("🚨 SurakshaNet AI Surveillance System")

# Buttons
start = st.button("🎥 Start Camera")
stop = st.button("🛑 Stop")

# Session state
if "run" not in st.session_state:
    st.session_state.run = False

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

if "risk_data" not in st.session_state:
    st.session_state.risk_data = ("LOW", 0, 0)

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False

# Placeholder
frame_placeholder = st.empty()

# Start camera
cap = cv2.VideoCapture(0)

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Camera not working")
        break

    # 🔥 Resize (faster processing)
    frame = cv2.resize(frame, (320, 240))

    st.session_state.frame_count += 1
    fc = st.session_state.frame_count

    # 🔥 Run detection every 5 frames (for speed)
    if fc % 5 == 0:
        st.session_state.risk_data = predict_risk(frame)

    risk, crowd_percent, people = st.session_state.risk_data

    # 🎨 Color logic
    color = (0, 255, 0)
    if risk == "MEDIUM":
        color = (0, 165, 255)
    elif risk == "HIGH":
        color = (0, 0, 255)

    # ✅ Updated text (REAL meaning now)
    text = f"{risk} ({crowd_percent}%) | People: {people}"

    cv2.putText(frame, text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2)

    frame_placeholder.image(frame, channels="BGR")

    time.sleep(0.001)

# Release camera
cap.release()