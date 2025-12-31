from flask import Flask, request, jsonify
# Flask        → bikin server backend
# request      → ambil data dari Flutter (gambar)
# jsonify      → kirim hasil ke Flutter dalam bentuk JSON

from flask_cors import CORS
# CORS → supaya backend bisa diakses dari Flutter / Web tanpa diblok browser

import cv2
# OpenCV → untuk decode dan mengolah gambar

import numpy as np
# NumPy → untuk hitung jarak, vektor, dan operasi angka

import base64
# base64 → decode gambar kalau dikirim dari Web (string base64)

import mediapipe as mp
# MediaPipe → library untuk deteksi wajah, mata, iris, dan pupil


# =========================
# INISIALISASI FLASK APP
# =========================

app = Flask(__name__)
# bikin aplikasi backend Flask

CORS(app)
# aktifkan CORS supaya Flutter boleh akses API ini


# =========================
# MEDIA PIPE FACE MESH
# =========================

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,              # hanya deteksi 1 wajah (lebih ringan)
    refine_landmarks=True,        # WAJIB agar iris & pupil terdeteksi
    min_detection_confidence=0.5, # minimal yakin 50% saat deteksi awal wajah
    min_tracking_confidence=0.5   # minimal yakin 50% saat tracking antar frame
)


# =========================
# LANDMARK INDEX (PATOKAN TITIK)
# =========================

# Pusat pupil dan lingkar iris kanan
RIGHT_PUPIL_CENTER = 468
RIGHT_IRIS_RING = [469, 470, 471, 472]

# Pusat pupil dan lingkar iris kiri
LEFT_PUPIL_CENTER = 473
LEFT_IRIS_RING = [474, 475, 476, 477]

# Sudut mata (inner & outer corner)
# Dipakai sebagai "anchor" supaya gerakan kepala tidak ikut terhitung
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

# Titik kelopak mata (buat deteksi mata terbuka/tertutup secara sederhana)
# (atas, bawah) untuk hitung "openness ratio" mirip EAR versi simple
LEFT_EYE_LID = (159, 145)     # atas kiri, bawah kiri
RIGHT_EYE_LID = (386, 374)    # atas kanan, bawah kanan


# =========================
# STATE (MENYIMPAN FRAME SEBELUMNYA)
# =========================

# Posisi pupil relatif frame sebelumnya (buat movement)
prev_left_rel = None
prev_right_rel = None

# HOLD hasil terakhir saat mata TERBUKA (biar saat merem hasil tidak loncat)
last_open_pupil = None
last_open_status = None


# =========================
# HELPER FUNCTIONS
# =========================

def decode_image(req):
    """
    Fungsi untuk membaca gambar dari request Flutter.
    Bisa menerima:
    1) multipart/form-data (Android / iOS)
    2) JSON base64 (Web)
    """

    # ----- MODE MULTIPART -----
    if "image" in req.files:
        img_bytes = req.files["image"].read()
        return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # ----- MODE BASE64 (WEB) -----
    if req.is_json:
        img64 = req.json.get("image")
        if img64 and img64.startswith("data:image"):
            _, encoded = img64.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    return None


def to_px(lm, idx, w, h):
    """Ubah koordinat landmark MediaPipe (0..1) jadi pixel asli."""
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)


def midpoint(a, b):
    """Hitung titik tengah dua titik."""
    return (a + b) / 2.0


def distance(a, b):
    """Jarak Euclidean dua titik."""
    return float(np.linalg.norm(a - b))


def calculate_movement(prev, now):
    """Movement antar frame."""
    if prev is None:
        return 0.0
    return distance(prev, now)


def eye_openness_ratio(lm, w, h, corners, lid_points):
    """
    Deteksi mata terbuka/tertutup (versi sederhana, mirip EAR):
    - horizontal = jarak sudut mata (corner1-corner2)
    - vertical   = jarak kelopak atas-bawah
    openness = vertical / horizontal

    Kalau nilainya kecil sekali → mata kemungkinan tertutup/merem.
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
    Fungsi utama backend:
    - Deteksi pupil kiri & kanan
    - Deteksi mata terbuka/tertutup
    - Kalau terbuka → hitung movement + update prev
    - Kalau tertutup → tetap bisa baca titik, tapi prev TIDAK diupdate (biar hold)
    """
    global prev_left_rel, prev_right_rel

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    # safety: iris butuh minimal 478 landmark
    if len(lm) < 478:
        return None

    # -------------------------
    # 1) Cek mata terbuka/tertutup dulu Eye aspec ratio
    # -------------------------
    left_open_ratio = eye_openness_ratio(lm, w, h, LEFT_EYE_CORNERS, LEFT_EYE_LID)
    right_open_ratio = eye_openness_ratio(lm, w, h, RIGHT_EYE_CORNERS, RIGHT_EYE_LID)

    # Semakin kecil threshold → semakin ketat dianggap "tertutup".
    EYE_CLOSED_THRESHOLD = 0.18

    left_is_closed = left_open_ratio < EYE_CLOSED_THRESHOLD
    right_is_closed = right_open_ratio < EYE_CLOSED_THRESHOLD

    # Kalau salah satu tertutup → anggap closed
    any_closed = left_is_closed or right_is_closed

    # -------------------------
    # 2) Hitung pupil center & radius 
    # -------------------------
    def calc(center_idx, ring):
        center = to_px(lm, center_idx, w, h)
        ring_pts = [to_px(lm, i, w, h) for i in ring]
        radius = np.mean([distance(center, p) for p in ring_pts])
        return center, float(radius)

    left_center, left_radius = calc(LEFT_PUPIL_CENTER, LEFT_IRIS_RING)
    right_center, right_radius = calc(RIGHT_PUPIL_CENTER, RIGHT_IRIS_RING)

    # ambil sudut mata untuk anchor/cornet
    l1 = to_px(lm, LEFT_EYE_CORNERS[0], w, h)
    l2 = to_px(lm, LEFT_EYE_CORNERS[1], w, h)
    r1 = to_px(lm, RIGHT_EYE_CORNERS[0], w, h)
    r2 = to_px(lm, RIGHT_EYE_CORNERS[1], w, h)

    left_anchor = midpoint(l1, l2)
    right_anchor = midpoint(r1, r2)

    # posisi pupil relatif terhadap mata (kompensasi gerak kepala)
    left_rel = left_center - left_anchor
    right_rel = right_center - right_anchor

    # -------------------------
    # 3) Kalau mata tertutup: jangan update prev → biar HOLD
    #    (movement kita set 0.0 supaya tidak bikin spike)
    # -------------------------
    if any_closed:
        left_norm = 0.0
        right_norm = 0.0

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
                "left": "closed" if left_is_closed else "open",
                "right": "closed" if right_is_closed else "open"
            },
            "any_closed": True
        }

    # -------------------------
    # 4) Kalau kedua mata terbuka: hitung movement + update prev
    # -------------------------
    left_move = calculate_movement(prev_left_rel, left_rel)
    right_move = calculate_movement(prev_right_rel, right_rel)

    left_eye_width = max(distance(l1, l2), 1e-6)
    right_eye_width = max(distance(r1, r2), 1e-6)

    left_norm = left_move / left_eye_width
    right_norm = right_move / right_eye_width

    # update prev karena mata terbuka (valid)
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
        "eye_state": {
            "left": "open",
            "right": "open"
        },
        "any_closed": False
    }


# =========================
# LOGIKA STATUS MATA (NORMAL / KEMUNGKINAN TUNANETRA)
# =========================

def eye_status(left_norm, right_norm):
    THRESHOLD = 0.02
    if left_norm < THRESHOLD and right_norm < THRESHOLD:
        return "kemungkinan_tunanetra"
    return "normal"


# =========================
# API ENDPOINT
# =========================

@app.route("/detect", methods=["POST"])
def detect():
    global last_open_pupil, last_open_status

    img = decode_image(request)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    result = extract_pupil(img)
    if result is None:
        return jsonify({"status": "pupil_not_found"})

    pupil = result["pupil"]
    any_closed = result["any_closed"]
    eye_state = result["eye_state"]

    # -------------------------
    # Kalau salah satu mata tertutup:
    # status = closed
    # hasil pupil & movement HOLD dari data terakhir mata terbuka
    # -------------------------
    if any_closed:
        # kalau sudah pernah ada data mata terbuka, pakai itu (HOLD)
        if last_open_pupil is not None:
            pupil_to_send = last_open_pupil
        else:
            # kalau belum ada data open (misal frame pertama langsung merem)
            # kirim data sekarang tapi movement 0 (supaya aman)
            pupil_to_send = pupil

        return jsonify({
            "status": "closed",
            "eye_state": eye_state,     # info tambahan (frontend boleh abaikan)
            "pupil": pupil_to_send,     # HOLD hasil terakhir
            "held_status": last_open_status  # optional info: status terakhir sebelum closed
        })

    # -------------------------
    # Kalau kedua mata terbuka:
    # hitung status normal / kemungkinan_tunanetra
    # update HOLD data
    # -------------------------
    status = eye_status(
        pupil["left"]["movement_norm"],
        pupil["right"]["movement_norm"]
    )

    # simpan sebagai hasil terakhir valid (untuk HOLD saat blink)
    last_open_pupil = pupil
    last_open_status = status

    return jsonify({
        "status": status,
        "eye_state": eye_state,   # info tambahan (frontend boleh abaikan)
        "pupil": pupil
    })


@app.route("/")
def home():
    return jsonify({"status": "Backend OK - Improved + Closed/Hold"})


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
