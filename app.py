from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter

import cv2
import numpy as np
import base64
import mediapipe as mp
import time


# =========================
# INIT APP
# =========================

app = Flask(__name__)
CORS(app)

# Counter untuk evaluasi (buat tabel detection rate)
status_counter = Counter()

# log untuk evaluasi blink/EAR (1 entry per request /detect)
blink_log = []
frame_id = 0


# =========================
# MEDIAPIPE FACE MESH
# =========================

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,        # wajib agar iris & pupil terdeteksi
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# LANDMARK INDEX
# =========================

RIGHT_PUPIL_CENTER = 468
RIGHT_IRIS_RING = [469, 470, 471, 472]

LEFT_PUPIL_CENTER = 473
LEFT_IRIS_RING = [474, 475, 476, 477]

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

LEFT_EYE_LID = (159, 145)
RIGHT_EYE_LID = (386, 374)


# =========================
# STATE (FRAME MEMORY)
# =========================

prev_left_rel = None
prev_right_rel = None

last_open_pupil = None
last_open_status = None


# =========================
# HELPER FUNCTIONS
# =========================

def decode_image(req):
    """
    Menerima gambar dari:
    1) multipart/form-data (mobile)
    2) JSON base64 (web)
    """
    # MODE MULTIPART
    if "image" in req.files:
        img_bytes = req.files["image"].read()
        return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # MODE BASE64
    if req.is_json:
        img64 = req.json.get("image")
        if img64:
            # support format "data:image/...;base64,xxxxx"
            if img64.startswith("data:image"):
                _, encoded = img64.split(",", 1)
            else:
                encoded = img64

            img_bytes = base64.b64decode(encoded)
            return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    return None


def to_px(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)


def midpoint(a, b):
    return (a + b) / 2.0


def distance(a, b):
    return float(np.linalg.norm(a - b))


def calculate_movement(prev, now):
    if prev is None:
        return 0.0
    return distance(prev, now)


def eye_openness_ratio(lm, w, h, corners, lid_points):
    """
    Openness ratio mirip EAR sederhana:
    openness = vertical / horizontal
    """
    c1 = to_px(lm, corners[0], w, h)
    c2 = to_px(lm, corners[1], w, h)
    top = to_px(lm, lid_points[0], w, h)
    bottom = to_px(lm, lid_points[1], w, h)

    horizontal = max(distance(c1, c2), 1e-6)
    vertical = distance(top, bottom)

    return float(vertical / horizontal)


# =========================
# CORE PROCESSING
# =========================

def extract_pupil(img):
    """
    Return dict jika face mesh ditemukan,
    return None jika tidak ada landmark wajah/iris.
    """
    global prev_left_rel, prev_right_rel

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    # refine_landmarks True -> butuh 478 landmark
    if len(lm) < 478:
        return None

    # 1) eye open/closed
    left_open_ratio = eye_openness_ratio(lm, w, h, LEFT_EYE_CORNERS, LEFT_EYE_LID)
    right_open_ratio = eye_openness_ratio(lm, w, h, RIGHT_EYE_CORNERS, RIGHT_EYE_LID)

    EYE_CLOSED_THRESHOLD = 0.18

    left_is_closed = left_open_ratio < EYE_CLOSED_THRESHOLD
    right_is_closed = right_open_ratio < EYE_CLOSED_THRESHOLD
    any_closed = left_is_closed or right_is_closed

    # 2) pupil center + radius
    def calc(center_idx, ring):
        center = to_px(lm, center_idx, w, h)
        ring_pts = [to_px(lm, i, w, h) for i in ring]
        radius = np.mean([distance(center, p) for p in ring_pts])
        return center, float(radius)

    left_center, left_radius = calc(LEFT_PUPIL_CENTER, LEFT_IRIS_RING)
    right_center, right_radius = calc(RIGHT_PUPIL_CENTER, RIGHT_IRIS_RING)

    # anchors
    l1 = to_px(lm, LEFT_EYE_CORNERS[0], w, h)
    l2 = to_px(lm, LEFT_EYE_CORNERS[1], w, h)
    r1 = to_px(lm, RIGHT_EYE_CORNERS[0], w, h)
    r2 = to_px(lm, RIGHT_EYE_CORNERS[1], w, h)

    left_anchor = midpoint(l1, l2)
    right_anchor = midpoint(r1, r2)

    # relative positions
    left_rel = left_center - left_anchor
    right_rel = right_center - right_anchor

    # 3) closed: do not update prev, movement_norm = 0
    if any_closed:
        return {
            "pupil": {
                "left": {
                    "center_x": float(left_center[0]),
                    "center_y": float(left_center[1]),
                    "movement_norm": 0.0,
                    "radius": float(left_radius)
                },
                "right": {
                    "center_x": float(right_center[0]),
                    "center_y": float(right_center[1]),
                    "movement_norm": 0.0,
                    "radius": float(right_radius)
                }
            },
            "eye_state": {
                "left": "closed" if left_is_closed else "open",
                "right": "closed" if right_is_closed else "open"
            },
            "any_closed": True,
            "ear": {
                "left": float(left_open_ratio),
                "right": float(right_open_ratio)
            }
        }

    # 4) open: movement + update prev
    left_move = calculate_movement(prev_left_rel, left_rel)
    right_move = calculate_movement(prev_right_rel, right_rel)

    left_eye_width = max(distance(l1, l2), 1e-6)
    right_eye_width = max(distance(r1, r2), 1e-6)

    left_norm = left_move / left_eye_width
    right_norm = right_move / right_eye_width

    prev_left_rel = left_rel
    prev_right_rel = right_rel

    return {
        "pupil": {
            "left": {
                "center_x": float(left_center[0]),
                "center_y": float(left_center[1]),
                "movement_norm": float(left_norm),
                "radius": float(left_radius)
            },
            "right": {
                "center_x": float(right_center[0]),
                "center_y": float(right_center[1]),
                "movement_norm": float(right_norm),
                "radius": float(right_radius)
            }
        },
        "eye_state": {"left": "open", "right": "open"},
        "any_closed": False,
        "ear": {
            "left": float(left_open_ratio),
            "right": float(right_open_ratio)
        }
    }


def eye_status(left_norm, right_norm):
    THRESHOLD = 0.02
    if left_norm < THRESHOLD and right_norm < THRESHOLD:
        return "kemungkinan_tunanetra"
    return "normal"


# =========================
# API ENDPOINTS
# =========================

@app.route("/detect", methods=["POST"])
def detect():
    global last_open_pupil, last_open_status, frame_id

    img = decode_image(request)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    result = extract_pupil(img)

    # pupil_not_found (face mesh / iris fail)
    if result is None:
        status_counter["pupil_not_found"] += 1

        frame_id += 1
        blink_log.append({
            "frame_id": frame_id,
            "t_ms": int(time.time() * 1000),
            "status": "pupil_not_found"
        })

        return jsonify({"status": "pupil_not_found"})

    pupil = result["pupil"]
    any_closed = result["any_closed"]
    eye_state = result["eye_state"]
    ear = result.get("ear")

    # closed + HOLD
    if any_closed:
        status_counter["closed"] += 1

        frame_id += 1
        blink_log.append({
            "frame_id": frame_id,
            "t_ms": int(time.time() * 1000),
            "status": "closed",
            "any_closed": True,
            "ear_left": ear["left"] if ear else None,
            "ear_right": ear["right"] if ear else None
        })

        if last_open_pupil is not None:
            pupil_to_send = last_open_pupil
        else:
            pupil_to_send = pupil

        return jsonify({
            "status": "closed",
            "eye_state": eye_state,
            "pupil": pupil_to_send,
            "held_status": last_open_status,
            "ear": ear
        })

    # open: compute status
    status = eye_status(
        pupil["left"]["movement_norm"],
        pupil["right"]["movement_norm"]
    )

    status_counter[status] += 1

    frame_id += 1
    blink_log.append({
        "frame_id": frame_id,
        "t_ms": int(time.time() * 1000),
        "status": status,
        "any_closed": False,
        "ear_left": ear["left"] if ear else None,
        "ear_right": ear["right"] if ear else None
    })

    last_open_pupil = pupil
    last_open_status = status

    return jsonify({
        "status": status,
        "eye_state": eye_state,
        "pupil": pupil,
        "ear": ear
    })


@app.route("/stats", methods=["GET"])
def stats():
    """
    Ambil statistik counter untuk isi tabel hasil.
    """
    total = sum(status_counter.values())
    return jsonify({
        "counter": dict(status_counter),
        "total_requests": total
    })


@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    """
    Reset counter sebelum uji kondisi baru.
    """
    status_counter.clear()
    return jsonify({"ok": True, "counter": dict(status_counter)})


@app.route("/reset_experiment", methods=["POST"])
def reset_experiment():
    """
    Reset semua yang relevan untuk eksperimen:
    - counter
    - blink_log
    - state memory (prev_*, last_open_*)
    """
    global prev_left_rel, prev_right_rel, last_open_pupil, last_open_status
    global frame_id

    status_counter.clear()
    blink_log.clear()
    frame_id = 0

    prev_left_rel = None
    prev_right_rel = None
    last_open_pupil = None
    last_open_status = None

    return jsonify({"ok": True})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend OK - Counter Enabled"})


@app.route("/stat", methods=["GET"])
def stat_alias():
    return stats()


@app.route("/export_blink_log", methods=["GET"])
def export_blink_log():
    return jsonify({
        "n": len(blink_log),
        "data": blink_log
    })


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)