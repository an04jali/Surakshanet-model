# 🛡️ SurakshaNet — AI-Powered Crowd Safety & Suspicious Activity Detection

> An AI-powered real-time surveillance and crowd safety system that detects people, analyzes crowd conditions, identifies suspicious activity, and sends instant Telegram alerts.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLOv8-Object%20Detection-green.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)

---

## 🚀 Live Demo

🌐 **Try SurakshaNet:**  
**[https://surakshanet-ai.streamlit.app/]**

The deployed application supports:

- 📹 Browser-based webcam monitoring
- 📤 Video upload and analysis
- 👥 Person detection
- 📊 Crowd density analysis
- ⚠️ Risk-level estimation
- 🧠 Suspicious activity detection
- 🔥 Heatmap visualization
- 🗺️ Zone-based monitoring
- 🔔 Alert history
- 📋 Analysis history
- 📱 Real-time Telegram notifications

---

# 📌 Project Overview

SurakshaNet is designed as an intelligent surveillance and crowd safety system.

Traditional surveillance systems often require continuous human monitoring. SurakshaNet attempts to automate part of this process by analyzing video streams and identifying potentially risky crowd situations.

The system processes video frames, detects people using a YOLO-based object detector, calculates crowd-related metrics, evaluates risk, and triggers alerts when predefined suspicious conditions are detected.

When a high-risk situation is detected, SurakshaNet can send an instant notification through Telegram.

---

# ✨ Key Features

## 👥 Person Detection

Uses **YOLOv8** to detect people in video frames.

The system tracks:

- Number of people detected
- Crowd density
- Movement-related information
- Risk level

---

## 📹 Browser Webcam Monitoring

SurakshaNet supports webcam monitoring directly through the browser.

The application uses:

```text
Streamlit WebRTC
       ↓
Browser Camera
       ↓
Video Frames
       ↓
YOLO Detection
       ↓
Risk Analysis
       ↓
Alert System

No cv2.VideoCapture(0) is required for the deployed application.

This allows recruiters and mentors to test the webcam directly from the deployed website.

📤 Video Upload & Analysis

Users can upload a video file and analyze it through the application.

Supported workflow:

Upload Video
      ↓
Read Video Frames
      ↓
Sample Frames
      ↓
YOLO Person Detection
      ↓
Crowd/Risk Analysis
      ↓
Display Results
      ↓
Save Analysis History

To reduce processing load, the uploaded video analysis processes every 4th frame.

This improves performance while maintaining useful detection results for surveillance demonstrations.

⚠️ Risk Detection

The system classifies the current situation into risk levels such as:

🟢 Low
🟡 Medium
🔴 High

Risk estimation considers crowd-related information and movement/activity indicators.

🚨 Telegram Alert System

When suspicious/high-risk activity is detected, SurakshaNet can send a Telegram notification.

Example:

🚨 SURAKSHANET ALERT 🚨

Source: crowd_test.mp4
Risk Level: High
People Detected: 15
Movement Score: 31

Suspicious activity detected.

Telegram alerts include a cooldown mechanism to prevent repeated notification spam.

🔔 Alert History

The application provides an Alert History section containing information such as:

Alert time
Source
Risk level
Number of people
Movement score
Behavior
Telegram status

Example:

🚨 Alert History

14:32:10
Source: crowd_test.mp4
Risk: High
People: 15
Movement: 0.31
Telegram: Sent
📊 Analysis History

SurakshaNet maintains analysis information for processed videos.

The history can include:

Video filename
Analysis timestamp
Total frames
Frames analyzed
Maximum people detected
Highest risk level
Number of alerts

Example:

Video	Frames	Analyzed	Max People	Risk	Alerts
crowd_test.mp4	480	120	15	High	3
market.mp4	600	150	8	Medium	1
🔥 Heatmap Visualization

SurakshaNet can visualize crowd activity using a heatmap.

This helps identify areas where people are concentrated or where activity is frequently detected.

🗺️ Zone Monitoring

The application provides zone-based monitoring to help divide the surveillance area into different regions.

This can be useful for identifying crowded or potentially risky areas.

🧠 AI / ML Components
YOLOv8

Used for real-time person detection.

Input Frame
     ↓
YOLOv8
     ↓
Person Bounding Boxes
     ↓
Person Count
Mini Transformer

A lightweight Transformer-based component is used for risk prediction using crowd count and movement-related features.

Input:

[count, movement]

Output:

Risk Score
Vision Transformer

A pretrained Vision Transformer is used as part of the activity analysis pipeline.

The system samples frames and aggregates model predictions to generate a stable activity classification.

🏗️ System Architecture
                    ┌─────────────────────┐
                    │     User / Browser  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          📹 Browser Webcam             📤 Video Upload
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                     Frame Processing
                               │
                               ▼
                         YOLOv8 Model
                               │
                               ▼
                    Person / Crowd Detection
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
          Crowd Analysis              Activity Analysis
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                         Risk Analysis
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
          Dashboard Results             Alert System
                                             │
                                             ▼
                                      📱 Telegram
🛠️ Technology Stack
Technology	Purpose
Python	Core programming language
Streamlit	Web application interface
Streamlit WebRTC	Browser webcam streaming
OpenCV	Video/frame processing
YOLOv8	Person/object detection
PyTorch	Deep learning
Torchvision	Vision models and preprocessing
Vision Transformer	Activity analysis
Mini Transformer	Risk prediction
NumPy	Numerical processing
Pandas	Data handling
Plotly	Data visualization
Telegram Bot API	Real-time alerts
📁 Project Structure
SurakshaNet/
│
├── app.py
├── alerts.py
├── model.py
├── tracker.py
├── mini_transformer.py
├── camera_test.py
│
├── requirements.txt
├── packages.txt
├── .gitignore
│
├── data/
│   └── ...
│
└── .streamlit/
    └── secrets.toml

secrets.toml contains private credentials and should never be committed to GitHub.

⚙️ Installation
1. Clone the Repository
git clone https://github.com/an04jali/Surakshanet-model.git
cd Surakshanet-model
2. Create Virtual Environment
python -m venv venv

Activate it on Windows:

venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Application
streamlit run app.py

The application will be available at:

http://localhost:8501
🔐 Telegram Configuration

Telegram credentials should be stored securely.

For local development:

.streamlit/
└── secrets.toml

Example:

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

For Streamlit Cloud, add these values through the application's Secrets settings.

⚠️ Never commit your Telegram bot token to GitHub.

☁️ Deployment

SurakshaNet is designed to run as a Streamlit web application.

Deployment flow:

GitHub Repository
       ↓
Streamlit Cloud
       ↓
Python Environment
       ↓
SurakshaNet Web App
       ↓
Browser Webcam / Video Upload

The application can be updated by pushing new changes to the main branch.

git add .
git commit -m "Update SurakshaNet"
git push origin main
🎥 Demo
📹 Video Demo

Add your project demonstration video here.

<!-- Replace VIDEO_URL with your YouTube/Loom/Google Drive video link -->


![SurakshaNet Dashboard](screenshots/dashboard.png)
📹 Live Webcam Detection

Add your video upload/analysis screenshot here.

![Video Analysis](screenshots/video-analysis.png)

🎯 Use Cases

SurakshaNet can be adapted for:

🏟️ Stadium crowd monitoring
🚉 Railway/metro stations
🛍️ Shopping malls
🎪 Events and festivals
🏫 Campus safety
🚪 Entry/exit monitoring
🏙️ Public-space surveillance
🚨 Early detection of unusual crowd activity
🔮 Future Improvements

Potential future improvements include:

Persistent cloud-based analysis history
Multi-camera support
Automatic incident snapshots
Email/SMS alerts
Cloud database integration
GPU-accelerated inference


👩‍💻 Author

Anjali

GitHub:
https://github.com/an04jali

⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub.

📜 License

Add your preferred license here.

For example:

MIT License

### One important recommendation

Since you're using this for **off-campus applications and recruiters**, I'd put the **Live Demo + screenshots + 1-minute demo video near the top**. Recruiters may not read the entire README.

Your top section should visually become:

```text
🛡️ SurakshaNet

AI-Powered Crowd Safety & Suspicious Activity Detection

[🚀 Live Demo]   [📂 GitHub]

📹 Webcam | 📤 Video Analysis | 🚨 Telegram Alerts | 📊 Analytics

[<img width="1838" height="899" alt="image" src="https://github.com/user-attachments/assets/b4934a60-bdfa-48ff-9358-9487289b2a6b" />
<img width="1643" height="885" alt="image" src="https://github.com/user-attachments/assets/2a639fe7-69e6-45da-ab3e-b6da4a2f1dbc" />
<img width="1843" height="904" alt="image" src="https://github.com/user-attachments/assets/0faf92ad-a1cd-45d1-a2c7-ed9fac1e640d" />
<img width="1847" height="842" alt="image" src="https://github.com/user-attachments/assets/22ef3ad5-8422-4fb9-b94f-b561c7e30974" />
<img width="1521" height="881" alt="image" src="https://github.com/user-attachments/assets/02437d17-1cd9-41bc-8f6e-d807859aa1e9" />
<img width="906" height="621" alt="image" src="https://github.com/user-attachments/assets/27613210-e526-4b1f-859b-260d19abb191" />
<img width="1401" height="799" alt="image" src="https://github.com/user-attachments/assets/cd244eba-56cb-4835-9468-5f4bafd2f932" />
<img width="751" height="958" alt="image" src="https://github.com/user-attachments/assets/b403f117-0ceb-486a-b787-5ee5b928ca09" />
]
