import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ================= LOAD MODELS =================

# Device detection (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[SurakshaNet] Loading YOLOv5 on {device}...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
yolo_model.eval()

print("[SurakshaNet] Loading MobileNetV3 (SE-Attention Backbone)...")
weights = MobileNet_V3_Small_Weights.DEFAULT
behavior_model = mobilenet_v3_small(weights=weights).to(device)
behavior_model.eval()

PERSON_CLASS = 0
CONF_THRESHOLD    =  0.35 # Thoda increase kiya for better accuracy
YOLO_INPUT_SIZE   = 416

# ================= TRANSFORM =================

crop_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ================= GLOBAL STATE =================

_frame_counter = 0
_last_behavior = "NORMAL"
_last_behavior_conf = 0.0
_last_counts = []

# ================= RISK LOGIC =================

def get_crowd_level(count, behavior="NORMAL", sensitivity=50):
    factor = sensitivity / 50.0
    
    # Dynamic thresholds based on sensitivity slider
    low_thresh = int(8 / factor)
    med_thresh = int(25 / factor)

    if count <= 2: return "Low"

    # Suspicious behavior crowd level ko "High" ki taraf push karta hai
    if behavior == "SUSPICIOUS":
        count += 5 

    if count <= low_thresh:
        return "Low"
    elif count <= med_thresh:
        return "Medium"
    else:
        return "High"

def get_crowd_percentage(count):
    # Mapping count to 0-100% scale
    return min(100, int((count / 40) * 100))

# ================= ZONES =================

def count_zones(frame_shape, boxes):
    h, w = frame_shape[:2]
    zones = {"A": 0, "B": 0, "C": 0, "D": 0}

    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cx < w//2 and cy < h//2: zones["A"] += 1
        elif cx >= w//2 and cy < h//2: zones["B"] += 1
        elif cx < w//2 and cy >= h//2: zones["C"] += 1
        else: zones["D"] += 1
    return zones

# ================= HEATMAP =================

def apply_heatmap(frame, boxes):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(heatmap, (cx, cy), 50, 1.0, -1)

    heatmap = cv2.GaussianBlur(heatmap, (61, 61), 0)
    heatmap = np.clip(heatmap / (heatmap.max() + 1e-5), 0, 1)
    
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

# ================= BEHAVIOR ANALYSIS (SE-ATTENTION SIM) =================

def analyze_behavior(frame, boxes):
    if not boxes: return "NORMAL", 0.0

    crops = []
    # Sirf top 3 detected people ko analyze karte hain to save FPS
    for (x1, y1, x2, y2) in boxes[:3]:
        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
        if crop.size == 0: continue
        
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(crop_transform(crop_rgb))

    if not crops: return "NORMAL", 0.0

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        outputs = behavior_model(batch)
        # Using Entropy as a proxy for 'Suspicious' behavior in this demo
        probs = torch.softmax(outputs, dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean().item()

    # Normalizing entropy to 0-1 scale
    sus_score = min(1.0, entropy / 5.5) 
    
    if sus_score > 0.75:
        return "SUSPICIOUS", round(sus_score * 100, 1)
    return "NORMAL", round((1 - sus_score) * 100, 1)

# ================= MAIN PROCESSING =================

def process_frame(frame, heatmap_on=True, zones_on=True, sensitivity=50, behavior_skip=8):
    global _frame_counter, _last_behavior, _last_behavior_conf, _last_counts

    _frame_counter += 1
    
    # Step 1: YOLO Detection
    results = yolo_model(frame, size=YOLO_INPUT_SIZE)
    detections = results.xyxy[0].cpu().numpy()

    person_boxes = []
    avg_conf = 0.0

    for *box, conf, cls in detections:
        if int(cls) == PERSON_CLASS and conf > CONF_THRESHOLD:
            person_boxes.append(list(map(int, box)))
            avg_conf += conf

    count = len(person_boxes)
    if count > 0: avg_conf /= count

    # Step 2: Smoothing Count
    _last_counts.append(count)
    if len(_last_counts) > 10: _last_counts.pop(0)
    smooth_count = int(np.mean(_last_counts))

    # Step 3: Behavior Analysis (Skip frames for performance)
    if _frame_counter % behavior_skip == 0:
        _last_behavior, _last_behavior_conf = analyze_behavior(frame, person_boxes)

    # Step 4: Density Index
    density = min(1.0, smooth_count / 30.0)

    # Step 5: Risk Assessment
    crowd_level = get_crowd_level(smooth_count, _last_behavior, sensitivity)
    crowd_pct = get_crowd_percentage(smooth_count)

    # Step 6: Visual Overlays
    display_frame = frame.copy()
    
    if zones_on:
        h, w = display_frame.shape[:2]
        cv2.line(display_frame, (w//2, 0), (w//2, h), (40, 40, 40), 1)
        cv2.line(display_frame, (0, h//2), (w, h//2), (40, 40, 40), 1)

    if heatmap_on and person_boxes:
        display_frame = apply_heatmap(display_frame, person_boxes)

    # Draw Bounding Boxes
    for (x1, y1, x2, y2) in person_boxes:
        color = (212, 182, 6) if crowd_level == "Low" else (11, 158, 245)
        if crowd_level == "High": color = (68, 68, 239)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

    zones = count_zones(frame.shape, person_boxes)

    return display_frame, smooth_count, density, crowd_level, crowd_pct, zones, _last_behavior, _last_behavior_conf