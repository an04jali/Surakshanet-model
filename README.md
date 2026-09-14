# 🛡️ SurakshaNet

### AI-Powered Crowd Monitoring & Risk Detection System

SurakshaNet is an AI-powered computer vision system designed to monitor
crowded environments through video analysis. It detects and tracks people,
analyzes crowd movement and density, and uses these visual patterns to
identify potentially unsafe or high-risk crowd situations.

The goal of SurakshaNet is to assist in early detection of abnormal crowd
conditions and support safer monitoring of public spaces, events, and
high-density environments.

---

## 🎯 Problem Statement

Large gatherings and crowded public spaces can become difficult to monitor
manually. Sudden changes in crowd density, movement patterns, or the
direction and speed of people can indicate potentially unsafe situations.

SurakshaNet addresses this problem by using computer vision and deep
learning to automatically analyze video footage and provide an AI-based
assessment of crowd conditions.

---

## 🚀 Key Features

- 👤 **Person Detection**
  - Detects people in video frames using YOLOv8.

- 🎯 **Person Tracking**
  - Tracks detected individuals across consecutive frames.

- 👥 **Crowd Monitoring**
  - Monitors the number and distribution of people in the scene.

- 📊 **Crowd Density Analysis**
  - Analyzes the concentration of people within the monitored area.

- 🏃 **Movement Analysis**
  - Analyzes crowd movement and changes in movement patterns.

- ⚠️ **Risk Assessment**
  - Uses extracted crowd characteristics to identify potentially
    unsafe crowd conditions.

- 🎥 **Video-Based Analysis**
  - Supports analysis of video input for automated monitoring.

---

## 🧠 AI & Computer Vision

SurakshaNet combines multiple computer vision and deep learning techniques
to analyze crowd behavior.

### YOLOv8

YOLOv8 is used as the primary object detection model to identify people
within video frames.

### Object Tracking

Tracking is used to maintain the identity/location of detected individuals
across consecutive frames and obtain movement-related information.

### Crowd Analysis

The tracking and detection information is used to derive crowd-related
characteristics such as:

- Number of people
- Crowd density
- Movement patterns
- Direction of movement
- Relative movement/speed

These characteristics are then used for risk assessment.

---

## 🔄 System Workflow

```text
                 ┌─────────────────────┐
                 │   Camera / Video    │
                 │       Input         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Frame Processing  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      YOLOv8         │
                 │   Person Detection  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Person Tracking   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Crowd Analysis    │
                 │ Density + Movement  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Risk Assessment   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Monitoring / Output  │
                 └─────────────────────┘
