from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter

import base64
import time

import cv2
import mediapipe as mp
import numpy as np


app = Flask(__name__)
CORS(app)


# =========================
# EXPERIMENT CONFIG
# =========================
# Pilihan:
# "none"    = tanpa preprocessing
# "retinex" = Retinex-Based Fast Algorithm
# "mclahe"  = Multiscale CLAHE / histogram excess-distribution
# "sci"     = Self-Calibrated Illumination sederhana / SCI-inspired
# "iagc"    = Illumination-Aware Gamma Correction sederhana / IAGC-inspired
ENHANCEMENT_METHOD = "iagc"


# =========================
# GLOBAL COUNTER & LOG
# =========================
status_counter = Counter()
blink_log = []
frame_id = 0


# =========================
# MEDIAPIPE CONFIG
# =========================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
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
# GLOBAL STATE
# =========================
prev_left_rel = None
prev_right_rel = None
last_open_pupil = None
last_open_status = None


# =========================
# IMAGE ENHANCEMENT METHODS
# =========================
def enhance_image_rbfa(img, sigma=50):
    """
    Retinex-Based Fast Algorithm sederhana.
    Digunakan untuk menormalkan pencahayaan dan memperjelas kontras citra.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_float = np.float32(v) + 1.0
    illumination = cv2.GaussianBlur(v_float, (0, 0), sigma)

    retinex = cv2.log(v_float) - cv2.log(illumination)
    v_enhanced = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)

    hsv_enhanced = cv2.merge([h, s, np.uint8(v_enhanced)])
    result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return result


def enhance_image_mclahe(img):
    """
    Multiscale Histogram Excess-Distribution berbasis CLAHE.

    Konsep:
    - Citra diubah ke LAB.
    - Peningkatan dilakukan pada kanal L/luminance.
    - CLAHE diterapkan pada beberapa ukuran tile.
    - Hasil dari beberapa skala digabungkan.
    - Citra diblend dengan luminance asli agar tidak terlalu over-enhanced.
    """

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    mean_l = np.mean(l_channel)

    if mean_l < 70:
        clip_limit = 3.5
    elif mean_l < 120:
        clip_limit = 3.0
    elif mean_l < 170:
        clip_limit = 2.5
    else:
        clip_limit = 2.0

    scales = [
        (4, 4),
        (8, 8),
        (16, 16)
    ]

    enhanced_layers = []

    for tile_size in scales:
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size
        )
        enhanced_l = clahe.apply(l_channel)
        enhanced_layers.append(enhanced_l.astype(np.float32))

    merged_l = (
        0.30 * enhanced_layers[0] +
        0.45 * enhanced_layers[1] +
        0.25 * enhanced_layers[2]
    )

    merged_l = np.clip(merged_l, 0, 255).astype(np.uint8)

    final_l = cv2.addWeighted(l_channel, 0.25, merged_l, 0.75, 0)

    final_lab = cv2.merge([final_l, a_channel, b_channel])
    result = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    return result


def enhance_image_sci(img):
    """
    Self-Calibrated Illumination sederhana / SCI-inspired.

    Catatan:
    Ini implementasi eksperimental berbasis prinsip SCI,
    bukan reproduksi penuh model deep learning SCI asli.
    """

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    illumination = np.max(rgb, axis=2)

    illumination_blur = cv2.GaussianBlur(
        illumination,
        (0, 0),
        sigmaX=15
    )

    illumination_blur = np.clip(illumination_blur, 0.05, 1.0)

    corrected = rgb / illumination_blur[:, :, np.newaxis]
    corrected = np.clip(corrected, 0, 1)

    mean_light = np.mean(illumination)

    if mean_light < 0.25:
        gamma = 0.65
    elif mean_light < 0.45:
        gamma = 0.75
    else:
        gamma = 0.90

    corrected = np.power(corrected, gamma)

    alpha = 0.70
    result = (alpha * corrected) + ((1 - alpha) * rgb)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def enhance_image_iagc(img):
    """
    Illumination-Aware Gamma Correction sederhana / IAGC-inspired.

    Konsep:
    - Menggunakan kanal luminance untuk membaca kondisi pencahayaan.
    - Gamma ditentukan secara adaptif berdasarkan tingkat gelap/terang citra.
    - Area gelap diperbaiki tanpa membuat area terang terlalu overexposed.

    Catatan:
    Ini implementasi eksperimental berbasis prinsip IAGC,
    bukan reproduksi penuh model deep learning IAGC asli.
    """

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_float = l_channel.astype(np.float32) / 255.0
    mean_l = np.mean(l_float)

    if mean_l < 0.25:
        gamma = 0.55
    elif mean_l < 0.40:
        gamma = 0.65
    elif mean_l < 0.55:
        gamma = 0.80
    else:
        gamma = 0.95

    enhanced_l = np.power(l_float, gamma)

    illumination_weight = 1.0 - l_float
    illumination_weight = cv2.GaussianBlur(
        illumination_weight,
        (0, 0),
        sigmaX=7
    )

    blended_l = (
        illumination_weight * enhanced_l +
        (1.0 - illumination_weight) * l_float
    )

    blended_l = np.clip(blended_l * 255, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )
    blended_l = clahe.apply(blended_l)

    final_lab = cv2.merge([blended_l, a_channel, b_channel])
    result = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    return result


def apply_enhancement(img):
    """
    Memilih metode preprocessing sebelum citra diproses MediaPipe.
    """

    if ENHANCEMENT_METHOD == "retinex":
        return enhance_image_rbfa(img, sigma=50)

    if ENHANCEMENT_METHOD == "mclahe":
        return enhance_image_mclahe(img)

    if ENHANCEMENT_METHOD == "sci":
        return enhance_image_sci(img)

    if ENHANCEMENT_METHOD == "iagc":
        return enhance_image_iagc(img)

    return img


# =========================
# HELPER FUNCTIONS
# =========================
def decode_image(req):
    """
    Membaca gambar dari multipart/form-data atau base64 JSON.
    """

    if "image" in req.files:
        img_bytes = req.files["image"].read()
        return cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

    if req.is_json:
        img64 = req.json.get("image")

        if img64:
            if img64.startswith("data:image"):
                _, encoded = img64.split(",", 1)
            else:
                encoded = img64

            img_bytes = base64.b64decode(encoded)
            return cv2.imdecode(
                np.frombuffer(img_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

    return None


def to_px(lm, idx, w, h):
    return np.array(
        [lm[idx].x * w, lm[idx].y * h],
        dtype=np.float32
    )


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


# =========================
# PUPIL EXTRACTION
# =========================
def extract_pupil(img):
    global prev_left_rel, prev_right_rel

    img = apply_enhancement(img)

    h, w = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    if len(lm) < 478:
        return None

    left_open_ratio = eye_openness_ratio(
        lm, w, h,
        LEFT_EYE_CORNERS,
        LEFT_EYE_LID
    )

    right_open_ratio = eye_openness_ratio(
        lm, w, h,
        RIGHT_EYE_CORNERS,
        RIGHT_EYE_LID
    )

    EYE_CLOSED_THRESHOLD = 0.18

    left_is_closed = left_open_ratio < EYE_CLOSED_THRESHOLD
    right_is_closed = right_open_ratio < EYE_CLOSED_THRESHOLD
    any_closed = left_is_closed or right_is_closed

    def calc(center_idx, ring):
        center = to_px(lm, center_idx, w, h)
        ring_pts = [to_px(lm, i, w, h) for i in ring]

        radius = np.mean([
            distance(center, p)
            for p in ring_pts
        ])

        return center, float(radius)

    left_center, left_radius = calc(
        LEFT_PUPIL_CENTER,
        LEFT_IRIS_RING
    )

    right_center, right_radius = calc(
        RIGHT_PUPIL_CENTER,
        RIGHT_IRIS_RING
    )

    l1 = to_px(lm, LEFT_EYE_CORNERS[0], w, h)
    l2 = to_px(lm, LEFT_EYE_CORNERS[1], w, h)

    r1 = to_px(lm, RIGHT_EYE_CORNERS[0], w, h)
    r2 = to_px(lm, RIGHT_EYE_CORNERS[1], w, h)

    left_anchor = midpoint(l1, l2)
    right_anchor = midpoint(r1, r2)

    left_eye_width = max(distance(l1, l2), 1e-6)
    right_eye_width = max(distance(r1, r2), 1e-6)

    left_rel_px = left_center - left_anchor
    right_rel_px = right_center - right_anchor

    left_rel_norm = left_rel_px / left_eye_width
    right_rel_norm = right_rel_px / right_eye_width

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
            },
            "geom": {
                "left_eye_width": float(left_eye_width),
                "right_eye_width": float(right_eye_width),
                "left_rel_px": [
                    float(left_rel_px[0]),
                    float(left_rel_px[1])
                ],
                "right_rel_px": [
                    float(right_rel_px[0]),
                    float(right_rel_px[1])
                ],
                "left_rel_norm": [
                    float(left_rel_norm[0]),
                    float(left_rel_norm[1])
                ],
                "right_rel_norm": [
                    float(right_rel_norm[0]),
                    float(right_rel_norm[1])
                ]
            }
        }

    left_move = calculate_movement(prev_left_rel, left_rel_px)
    right_move = calculate_movement(prev_right_rel, right_rel_px)

    left_norm = left_move / left_eye_width
    right_norm = right_move / right_eye_width

    prev_left_rel = left_rel_px
    prev_right_rel = right_rel_px

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
        "eye_state": {
            "left": "open",
            "right": "open"
        },
        "any_closed": False,
        "ear": {
            "left": float(left_open_ratio),
            "right": float(right_open_ratio)
        },
        "geom": {
            "left_eye_width": float(left_eye_width),
            "right_eye_width": float(right_eye_width),
            "left_rel_px": [
                float(left_rel_px[0]),
                float(left_rel_px[1])
            ],
            "right_rel_px": [
                float(right_rel_px[0]),
                float(right_rel_px[1])
            ],
            "left_rel_norm": [
                float(left_rel_norm[0]),
                float(left_rel_norm[1])
            ],
            "right_rel_norm": [
                float(right_rel_norm[0]),
                float(right_rel_norm[1])
            ]
        }
    }


# =========================
# STATUS CLASSIFICATION
# =========================
def eye_status(left_norm, right_norm):
    THRESHOLD = 0.02

    if left_norm < THRESHOLD and right_norm < THRESHOLD:
        return "kemungkinan_tunanetra"

    return "normal"


# =========================
# ROUTES
# =========================
@app.route("/detect", methods=["POST"])
def detect():
    global last_open_pupil, last_open_status, frame_id

    img = decode_image(request)

    if img is None:
        return jsonify({
            "error": "Invalid image",
            "enhancement_method": ENHANCEMENT_METHOD
        }), 400

    result = extract_pupil(img)

    if result is None:
        status_counter["pupil_not_found"] += 1
        frame_id += 1

        blink_log.append({
            "frame_id": frame_id,
            "t_ms": int(time.time() * 1000),
            "status": "pupil_not_found",
            "enhancement_method": ENHANCEMENT_METHOD
        })

        return jsonify({
            "status": "pupil_not_found",
            "enhancement_method": ENHANCEMENT_METHOD
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
            "geom": geom,
            "enhancement_method": ENHANCEMENT_METHOD
        })

        pupil_to_send = last_open_pupil if last_open_pupil is not None else pupil

        return jsonify({
            "status": "closed",
            "eye_state": eye_state,
            "pupil": pupil_to_send,
            "held_status": last_open_status,
            "ear": ear,
            "geom": geom,
            "enhancement_method": ENHANCEMENT_METHOD
        })

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
        "ear_right": ear["right"] if ear else None,
        "geom": geom,
        "enhancement_method": ENHANCEMENT_METHOD
    })

    last_open_pupil = pupil
    last_open_status = status

    return jsonify({
        "status": status,
        "eye_state": eye_state,
        "pupil": pupil,
        "ear": ear,
        "geom": geom,
        "enhancement_method": ENHANCEMENT_METHOD
    })


@app.route("/stats", methods=["GET"])
def stats():
    total = sum(status_counter.values())

    return jsonify({
        "counter": dict(status_counter),
        "total_requests": total,
        "enhancement_method": ENHANCEMENT_METHOD
    })


@app.route("/reset_experiment", methods=["POST"])
def reset_experiment():
    global prev_left_rel, prev_right_rel
    global last_open_pupil, last_open_status
    global frame_id

    status_counter.clear()
    blink_log.clear()

    frame_id = 0
    prev_left_rel = None
    prev_right_rel = None
    last_open_pupil = None
    last_open_status = None

    return jsonify({
        "ok": True,
        "enhancement_method": ENHANCEMENT_METHOD
    })


@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    status_counter.clear()

    return jsonify({
        "ok": True,
        "counter": dict(status_counter),
        "enhancement_method": ENHANCEMENT_METHOD
    })


@app.route("/export_blink_log", methods=["GET"])
def export_blink_log():
    return jsonify({
        "n": len(blink_log),
        "enhancement_method": ENHANCEMENT_METHOD,
        "data": blink_log
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Backend OK - Counter Enabled",
        "enhancement_method": ENHANCEMENT_METHOD
    })


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )