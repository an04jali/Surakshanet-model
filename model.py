import cv2
import torch
import numpy as np
from tracker import CentroidTracker
from ultralytics import YOLO


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[SurakshaNet] Loading YOLOv8 person detector on {device}...")


# ============================================================
# YOLO MODEL
# ============================================================

yolo_model = YOLO("yolov8n.pt")

print("[SurakshaNet] YOLOv8 person detector loaded successfully.")


# ============================================================
# CONFIGURATION
# ============================================================

PERSON_CLASS = 0

CONF_THRESHOLD = 0.25

YOLO_INPUT_SIZE = 640

# Maximum distance a person can move between
# consecutive frames for matching
MAX_MATCH_DISTANCE = 100


# ============================================================
# GLOBAL STATE
# ============================================================

_frame_counter = 0

# Previous frame person centers
_previous_centers = []

# Smoothed crowd counts
_last_counts = []

# Previous motion values
_last_avg_speed = 0.0
_last_movement_intensity = 0.0
_last_direction_variance = 0.0

# Current behaviour
_last_behavior = "NORMAL"
_last_behavior_conf = 0.0


# ============================================================
# BEHAVIOUR TEMPORAL STABILITY
# ============================================================

_suspicious_streak = 0
_normal_streak = 0

# Smoothed motion values
_smooth_speed = 0.0
_smooth_movement = 0.0
_smooth_direction = 0.0


# ============================================================
# BASIC UTILITY FUNCTIONS
# ============================================================

def center_of_box(box):
    """
    Calculate center point of bounding box.

    box = [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = box

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return (cx, cy)


def calculate_distance(p1, p2):
    """
    Euclidean distance between two points.
    """

    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# ANONYMOUS MOTION TRACKING
# ============================================================

def calculate_motion(current_centers):
    """
    Estimate crowd movement between consecutive frames.

    IMPORTANT:
    This does NOT identify people.

    It only compares anonymous positions between
    consecutive frames to estimate movement.
    """

    global _previous_centers

    if not current_centers or not _previous_centers:

        _previous_centers = current_centers.copy()

        return {
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "movement_intensity": 0.0,
            "direction_variance": 0.0,
            "moving_people": 0
        }

    movements = []

    used_previous = set()

    # Match every current point with the closest
    # previous point.
    for current in current_centers:

        best_distance = float("inf")
        best_index = None

        for i, previous in enumerate(_previous_centers):

            if i in used_previous:
                continue

            distance = calculate_distance(
                current,
                previous
            )

            if distance < best_distance:

                best_distance = distance
                best_index = i

        if (
            best_index is not None
            and best_distance <= MAX_MATCH_DISTANCE
        ):

            previous = _previous_centers[best_index]

            dx = current[0] - previous[0]
            dy = current[1] - previous[1]

            speed = np.sqrt(dx ** 2 + dy ** 2)

            # Ignore detector/bounding-box jitter.
            if speed < 5:
                speed = 0.0

            if speed > 0:

                angle = np.degrees(
                    np.arctan2(dy, dx)
                )

                movements.append(
                    {
                        "speed": float(speed),
                        "angle": float(angle),
                        "dx": float(dx),
                        "dy": float(dy)
                    }
                )

            used_previous.add(best_index)

    # Update previous positions
    _previous_centers = current_centers.copy()

    if not movements:

        return {
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "movement_intensity": 0.0,
            "direction_variance": 0.0,
            "moving_people": 0
        }

    speeds = [
        m["speed"]
        for m in movements
    ]

    angles = [
        m["angle"]
        for m in movements
    ]

    avg_speed = float(np.mean(speeds))

    max_speed = float(np.max(speeds))

    moving_people = len(movements)

    # Percentage of detected people who are moving
    movement_intensity = (
        moving_people /
        max(len(current_centers), 1)
    )

    # Circular variance for directions.
    # 0 → people moving mostly in the same direction
    # 1 → highly inconsistent directions
    angles_rad = np.radians(angles)

    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))

    direction_consistency = np.sqrt(
        mean_sin ** 2 +
        mean_cos ** 2
    )

    direction_variance = 1 - direction_consistency

    return {
        "avg_speed": round(avg_speed, 2),
        "max_speed": round(max_speed, 2),
        "movement_intensity": round(
            float(movement_intensity),
            3
        ),
        "direction_variance": round(
            float(direction_variance),
            3
        ),
        "moving_people": moving_people
    }


# ============================================================
# CROWD DENSITY
# ============================================================

def get_crowd_percentage(count):

    return min(
        100,
        int((count / 40) * 100)
    )


def calculate_density(count, frame_shape):

    h, w = frame_shape[:2]

    frame_area = h * w

    # Approximate people-per-frame-area index
    density = (
        count * 50000
    ) / max(frame_area, 1)

    return min(
        1.0,
        round(density, 3)
    )


# ============================================================
# ZONE ANALYSIS
# ============================================================

def count_zones(frame_shape, boxes):

    h, w = frame_shape[:2]

    zones = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0
    }

    for box in boxes:

        cx, cy = center_of_box(box)

        if cx < w // 2 and cy < h // 2:

            zones["A"] += 1

        elif cx >= w // 2 and cy < h // 2:

            zones["B"] += 1

        elif cx < w // 2 and cy >= h // 2:

            zones["C"] += 1

        else:

            zones["D"] += 1

    return zones


# ============================================================
# HEATMAP
# ============================================================

def apply_heatmap(frame, boxes):

    heatmap = np.zeros(
        (
            frame.shape[0],
            frame.shape[1]
        ),
        dtype=np.float32
    )

    for box in boxes:

        cx, cy = center_of_box(box)

        cv2.circle(
            heatmap,
            (cx, cy),
            50,
            1.0,
            -1
        )

    heatmap = cv2.GaussianBlur(
        heatmap,
        (61, 61),
        0
    )

    max_value = heatmap.max()

    if max_value > 0:

        heatmap = heatmap / max_value

    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    return cv2.addWeighted(
        frame,
        0.7,
        heatmap_color,
        0.3,
        0
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    count,
    density,
    avg_speed,
    movement_intensity,
    direction_variance,
    sensitivity=50
):
    """
    Calculate interpretable crowd risk score.

    Score:
        0 - 39   LOW
        40 - 69  MEDIUM
        70 - 100 HIGH
    """

    # --------------------------------------------------------
    # 1. Density contribution
    # --------------------------------------------------------

    density_score = min(
        100,
        density * 100
    )

    # --------------------------------------------------------
    # 2. Crowd count contribution
    # --------------------------------------------------------

    count_score = min(
        100,
        (count / 30) * 100
    )

    # --------------------------------------------------------
    # 3. Speed contribution
    # --------------------------------------------------------

    speed_score = min(
        100,
        (avg_speed / 40) * 100
    )

    # --------------------------------------------------------
    # 4. Movement contribution
    # --------------------------------------------------------

    movement_score = movement_intensity * 100

    # --------------------------------------------------------
    # 5. Direction inconsistency
    # --------------------------------------------------------

    direction_score = direction_variance * 100

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    score = (
        density_score * 0.30 +
        count_score * 0.20 +
        speed_score * 0.20 +
        movement_score * 0.15 +
        direction_score * 0.15
    )

    # Sensitivity adjustment
    sensitivity_factor = sensitivity / 50.0

    score = score * sensitivity_factor

    score = min(
        100,
        max(0, score)
    )

    return round(score, 1)


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(risk_score):

    if risk_score < 40:

        return "Low"

    elif risk_score < 70:

        return "Medium"

    else:

        return "High"


# ============================================================
# BEHAVIOUR INTERPRETATION
# ============================================================

def classify_behavior(
    avg_speed,
    movement_intensity,
    direction_variance,
    risk_score
):
    """
    Conservative crowd behaviour classification.

    This is a heuristic behaviour layer, not a trained
    action-recognition model. A single noisy frame should
    not be treated as suspicious.
    """

    # Stationary / very slow person -> NORMAL.
    if avg_speed < 12:

        return "NORMAL", 95.0

    # High movement with disordered directions.
    if (
        avg_speed >= 28
        and movement_intensity >= 0.50
        and direction_variance >= 0.45
    ):

        confidence = min(
            95.0,
            65.0 + risk_score * 0.30
        )

        return "SUSPICIOUS", round(
            confidence,
            1
        )

    # Very strong crowd movement.
    if (
        avg_speed >= 35
        and movement_intensity >= 0.60
    ):

        confidence = min(
            93.0,
            60.0 + risk_score * 0.30
        )

        return "SUSPICIOUS", round(
            confidence,
            1
        )

    return "NORMAL", 90.0


# ============================================================
# MAIN FRAME PROCESSING
# ============================================================

def process_frame(
    frame,
    heatmap_on=True,
    zones_on=True,
    sensitivity=50,
    behavior_skip=1
):

    global _frame_counter
    global _last_behavior
    global _last_behavior_conf
    global _last_counts
    global _last_avg_speed
    global _last_movement_intensity
    global _last_direction_variance
    global _suspicious_streak
    global _normal_streak
    global _smooth_speed
    global _smooth_movement
    global _smooth_direction

    _frame_counter += 1

    # --------------------------------------------------------
    # STEP 1: YOLOv8 DETECTION
    # --------------------------------------------------------

    results = yolo_model(
        frame,
        imgsz=YOLO_INPUT_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    boxes = results[0].boxes

    detections = []

    if boxes is not None:

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for box, conf, cls in zip(
            xyxy,
            confs,
            classes
        ):

            detections.append([
                *box,
                conf,
                cls
            ])

    # --------------------------------------------------------
    # STEP 2: PERSON DETECTIONS
    # --------------------------------------------------------

    person_boxes = []

    avg_conf = 0.0

    for *box, conf, cls in detections:

        if (
            int(cls) == PERSON_CLASS
            and float(conf) > CONF_THRESHOLD
        ):

            clean_box = list(
                map(int, box)
            )

            person_boxes.append(
                clean_box
            )

            avg_conf += float(conf)

    count = len(person_boxes)

    if count > 0:

        avg_conf /= count

    # --------------------------------------------------------
    # STEP 3: SMOOTH CROWD COUNT
    # --------------------------------------------------------

    _last_counts.append(count)

    if len(_last_counts) > 10:

        _last_counts.pop(0)

    smooth_count = int(
        np.mean(_last_counts)
    )

    # --------------------------------------------------------
    # STEP 4: ANONYMOUS CENTERS
    # --------------------------------------------------------

    current_centers = [
        center_of_box(box)
        for box in person_boxes
    ]

    # --------------------------------------------------------
    # STEP 5: MOTION ANALYSIS
    # --------------------------------------------------------

    motion = calculate_motion(
        current_centers
    )

    avg_speed = motion["avg_speed"]

    max_speed = motion["max_speed"]

    movement_intensity = motion[
        "movement_intensity"
    ]

    direction_variance = motion[
        "direction_variance"
    ]

    _last_avg_speed = avg_speed
    _last_movement_intensity = movement_intensity
    _last_direction_variance = direction_variance

    # --------------------------------------------------------
    # STEP 6: DENSITY
    # --------------------------------------------------------

    density = calculate_density(
        smooth_count,
        frame.shape
    )

    # --------------------------------------------------------
    # STEP 7: ZONES
    # --------------------------------------------------------

    zones = count_zones(
        frame.shape,
        person_boxes
    )

    # --------------------------------------------------------
    # STEP 8: RISK SCORE
    # --------------------------------------------------------

    risk_score = calculate_risk_score(
        smooth_count,
        density,
        avg_speed,
        movement_intensity,
        direction_variance,
        sensitivity
    )

    crowd_level = get_risk_level(
        risk_score
    )

    crowd_pct = get_crowd_percentage(
        smooth_count
    )

    # --------------------------------------------------------
    # STEP 9: TEMPORAL MOTION SMOOTHING
    # --------------------------------------------------------

    speed_alpha = 0.20
    movement_alpha = 0.20
    direction_alpha = 0.20

    _smooth_speed = (
        speed_alpha * avg_speed
        + (1.0 - speed_alpha) * _smooth_speed
    )

    _smooth_movement = (
        movement_alpha * movement_intensity
        + (1.0 - movement_alpha) * _smooth_movement
    )

    _smooth_direction = (
        direction_alpha * direction_variance
        + (1.0 - direction_alpha) * _smooth_direction
    )

    # --------------------------------------------------------
    # STEP 10: BEHAVIOUR CANDIDATE
    # --------------------------------------------------------

    candidate_behavior, candidate_conf = classify_behavior(
        _smooth_speed,
        _smooth_movement,
        _smooth_direction,
        risk_score
    )

    # --------------------------------------------------------
    # STEP 11: TEMPORAL CONFIRMATION
    # --------------------------------------------------------

    if candidate_behavior == "SUSPICIOUS":

        _suspicious_streak += 1
        _normal_streak = 0

        # Require 5 consecutive suspicious frames.
        if _suspicious_streak >= 5:

            _last_behavior = "SUSPICIOUS"
            _last_behavior_conf = candidate_conf

    else:

        _normal_streak += 1
        _suspicious_streak = 0

        # Return to NORMAL quickly when movement stops.
        if _normal_streak >= 2:

            _last_behavior = "NORMAL"
            _last_behavior_conf = candidate_conf

    # --------------------------------------------------------
    # STEP 12: VISUALIZATION
    # --------------------------------------------------------

    display_frame = frame.copy()

    # Zones
    if zones_on:

        h, w = display_frame.shape[:2]

        cv2.line(
            display_frame,
            (w // 2, 0),
            (w // 2, h),
            (40, 40, 40),
            1
        )

        cv2.line(
            display_frame,
            (0, h // 2),
            (w, h // 2),
            (40, 40, 40),
            1
        )

    # Heatmap
    if heatmap_on and person_boxes:

        display_frame = apply_heatmap(
            display_frame,
            person_boxes
        )

    # Bounding boxes
    for box in person_boxes:

        x1, y1, x2, y2 = box

        if crowd_level == "Low":

            color = (0, 200, 100)

        elif crowd_level == "Medium":

            color = (0, 165, 255)

        else:

            color = (0, 0, 255)

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

    # --------------------------------------------------------
    # DISPLAY INFORMATION ON VIDEO
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"People: {smooth_count}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Risk: {crowd_level} ({risk_score}%)",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Speed: {avg_speed:.1f}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Movement: {movement_intensity:.2f}",
        (15, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return (
        display_frame,
        smooth_count,
        density,
        crowd_level,
        crowd_pct,
        zones,
        _last_behavior,
        _last_behavior_conf
    )