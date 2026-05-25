from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter

import cv2
import numpy as np
import base64
import mediapipe as mp
import time
import tensorflow as tf  # <-- LIBRARY AI BARU

app = Flask(__name__)
CORS(app)

status_counter = Counter()

# log eksperimen (1 entry per request /detect)
blink_log = []
frame_id = 0

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

RIGHT_PUPIL_CENTER = 468
RIGHT_IRIS_RING = [469, 470, 471, 472]
LEFT_PUPIL_CENTER = 473
LEFT_IRIS_RING = [474, 475, 476, 477]

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

LEFT_EYE_LID = (159, 145)
RIGHT_EYE_LID = (386, 374)

prev_left_rel = None
prev_right_rel = None
last_open_pupil = None
last_open_status = None

# ====================================================================
# INISIALISASI OTAK AI (TENSORFLOW LITE)
# Pastikan file "model_wajah.tflite" ada di folder yang sama!
# ====================================================================
try:
    interpreter = tf.lite.Interpreter(model_path="model_wajah.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    TFLITE_READY = True
except Exception as e:
    print(f"WARNING: Gagal memuat model_wajah.tflite! Error: {e}")
    TFLITE_READY = False

def recognize_face(img):
    if not TFLITE_READY:
        return -1, 0.0
    
    try:
        # Resize gambar sesuai ukuran training di Colab (128x128)
        img_resized = cv2.resize(img, (128, 128))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Siapkan format datanya (float32)
        input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32)
        
        # Suruh AI menebak wajahnya
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Ambil index dengan tingkat keyakinan tertinggi
        predicted_index = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0]))
        
        return int(predicted_index), confidence
    except Exception as e:
        print(f"Error deteksi wajah: {e}")
        return -1, 0.0
# ====================================================================

def decode_image(req):
    if "image" in req.files:
        img_bytes = req.files["image"].read()
        return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if req.is_json:
        img64 = req.json.get("image")
        if img64:
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
    c1 = to_px(lm, corners[0], w, h)
    c2 = to_px(lm, corners[1], w, h)
    top = to_px(lm, lid_points[0], w, h)
    bottom = to_px(lm, lid_points[1], w, h)

    horizontal = max(distance(c1, c2), 1e-6)
    vertical = distance(top, bottom)
    return float(vertical / horizontal)

def extract_pupil(img):
    global prev_left_rel, prev_right_rel

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    if len(lm) < 478:
        return None

    # eye open/closed + EAR
    left_open_ratio = eye_openness_ratio(lm, w, h, LEFT_EYE_CORNERS, LEFT_EYE_LID)
    right_open_ratio = eye_openness_ratio(lm, w, h, RIGHT_EYE_CORNERS, RIGHT_EYE_LID)

    EYE_CLOSED_THRESHOLD = 0.18
    left_is_closed = left_open_ratio < EYE_CLOSED_THRESHOLD
    right_is_closed = right_open_ratio < EYE_CLOSED_THRESHOLD
    any_closed = left_is_closed or right_is_closed

    def calc(center_idx, ring):
        center = to_px(lm, center_idx, w, h)
        ring_pts = [to_px(lm, i, w, h) for i in ring]
        radius = np.mean([distance(center, p) for p in ring_pts])
        return center, float(radius)

    left_center, left_radius = calc(LEFT_PUPIL_CENTER, LEFT_IRIS_RING)
    right_center, right_radius = calc(RIGHT_PUPIL_CENTER, RIGHT_IRIS_RING)

    # corners & anchors
    l1 = to_px(lm, LEFT_EYE_CORNERS[0], w, h)
    l2 = to_px(lm, LEFT_EYE_CORNERS[1], w, h)
    r1 = to_px(lm, RIGHT_EYE_CORNERS[0], w, h)
    r2 = to_px(lm, RIGHT_EYE_CORNERS[1], w, h)

    left_anchor = midpoint(l1, l2)
    right_anchor = midpoint(r1, r2)

    left_eye_width = max(distance(l1, l2), 1e-6)
    right_eye_width = max(distance(r1, r2), 1e-6)

    # relative position (pixel)
    left_rel_px = left_center - left_anchor
    right_rel_px = right_center - right_anchor

    # relative position normalized
    left_rel_norm = left_rel_px / left_eye_width
    right_rel_norm = right_rel_px / right_eye_width

    # movement_norm hanya update saat open
    if any_closed:
        return {
            "pupil": {
                "left": {"center_x": float(left_center[0]), "center_y": float(left_center[1]), "movement_norm": 0.0, "radius": float(left_radius)},
                "right": {"center_x": float(right_center[0]), "center_y": float(right_center[1]), "movement_norm": 0.0, "radius": float(right_radius)}
            },
            "eye_state": {"left": "closed" if left_is_closed else "open", "right": "closed" if right_is_closed else "open"},
            "any_closed": True,
            "ear": {"left": float(left_open_ratio), "right": float(right_open_ratio)},
            "geom": {
                "left_eye_width": float(left_eye_width),
                "right_eye_width": float(right_eye_width),
                "left_rel_px": [float(left_rel_px[0]), float(left_rel_px[1])],
                "right_rel_px": [float(right_rel_px[0]), float(right_rel_px[1])],
                "left_rel_norm": [float(left_rel_norm[0]), float(left_rel_norm[1])],
                "right_rel_norm": [float(right_rel_norm[0]), float(right_rel_norm[1])]
            }
        }

    # open
    left_move = calculate_movement(prev_left_rel, left_rel_px)
    right_move = calculate_movement(prev_right_rel, right_rel_px)

    left_norm = left_move / left_eye_width
    right_norm = right_move / right_eye_width

    prev_left_rel = left_rel_px
    prev_right_rel = right_rel_px

    return {
        "pupil": {
            "left": {"center_x": float(left_center[0]), "center_y": float(left_center[1]), "movement_norm": float(left_norm), "radius": float(left_radius)},
            "right": {"center_x": float(right_center[0]), "center_y": float(right_center[1]), "movement_norm": float(right_norm), "radius": float(right_radius)}
        },
        "eye_state": {"left": "open", "right": "open"},
        "any_closed": False,
        "ear": {"left": float(left_open_ratio), "right": float(right_open_ratio)},
        "geom": {
            "left_eye_width": float(left_eye_width),
            "right_eye_width": float(right_eye_width),
            "left_rel_px": [float(left_rel_px[0]), float(left_rel_px[1])],
            "right_rel_px": [float(right_rel_px[0]), float(right_rel_px[1])],
            "left_rel_norm": [float(left_rel_norm[0]), float(left_rel_norm[1])],
            "right_rel_norm": [float(right_rel_norm[0]), float(right_rel_norm[1])]
        }
    }

def eye_status(left_norm, right_norm):
    THRESHOLD = 0.02
    if left_norm < THRESHOLD and right_norm < THRESHOLD:
        return "kemungkinan_tunanetra"
    return "normal"

@app.route("/detect", methods=["POST"])
def detect():
    global last_open_pupil, last_open_status, frame_id

    img = decode_image(request)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # 1. Jalankan deteksi pupil (kode asli lu)
    result = extract_pupil(img)
    
    # 2. JALANKAN DETEKSI WAJAH (TAMBAHAN AI)
    face_id, face_confidence = recognize_face(img)

    if result is None:
        status_counter["pupil_not_found"] += 1
        frame_id += 1
        blink_log.append({
            "frame_id": frame_id,
            "t_ms": int(time.time() * 1000),
            "status": "pupil_not_found"
        })
        return jsonify({
            "status": "pupil_not_found",
            "face_id_terdeteksi": face_id,
            "face_confidence": face_confidence
        })

    pupil = result["pupil"]
    any_closed = result["any_closed"]
    eye_state = result["eye_state"]
    ear = result.get("ear")
    geom = result.get("geom")

    if any_closed:
        status_counter["closed"] += 1
        frame_id += 1
        blink_log.append({
            "frame_id": frame_id,
            "t_ms": int(time.time() * 1000),
            "status": "closed",
            "any_closed": True,
            "ear_left": ear["left"] if ear else None,
            "ear_right": ear["right"] if ear else None,
            "geom": geom
        })

        pupil_to_send = last_open_pupil if last_open_pupil is not None else pupil

        return jsonify({
            "status": "closed",
            "eye_state": eye_state,
            "pupil": pupil_to_send,
            "held_status": last_open_status,
            "ear": ear,
            "geom": geom,
            "face_id_terdeteksi": face_id,
            "face_confidence": face_confidence
        })

    status = eye_status(pupil["left"]["movement_norm"], pupil["right"]["movement_norm"])
    status_counter[status] += 1

    frame_id += 1
    blink_log.append({
        "frame_id": frame_id,
        "t_ms": int(time.time() * 1000),
        "status": status,
        "any_closed": False,
        "ear_left": ear["left"] if ear else None,
        "ear_right": ear["right"] if ear else None,
        "geom": geom
    })

    last_open_pupil = pupil
    last_open_status = status

    return jsonify({
        "status": status,
        "eye_state": eye_state,
        "pupil": pupil,
        "ear": ear,
        "geom": geom,
        "face_id_terdeteksi": face_id,
        "face_confidence": face_confidence
    })

@app.route("/stats", methods=["GET"])
def stats():
    total = sum(status_counter.values())
    return jsonify({"counter": dict(status_counter), "total_requests": total})

@app.route("/reset_experiment", methods=["POST"])
def reset_experiment():
    global prev_left_rel, prev_right_rel, last_open_pupil, last_open_status, frame_id
    status_counter.clear()
    blink_log.clear()
    frame_id = 0
    prev_left_rel = None
    prev_right_rel = None
    last_open_pupil = None
    last_open_status = None
    return jsonify({"ok": True})

# OPTIONAL: biar gak 404 kalau kamu kebiasaan pakai /reset_stats
@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    status_counter.clear()
    return jsonify({"ok": True, "counter": dict(status_counter)})

@app.route("/export_blink_log", methods=["GET"])
def export_blink_log():
    return jsonify({"n": len(blink_log), "data": blink_log})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend OK - Counter Enabled"})

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)