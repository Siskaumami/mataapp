# ============================================================
#  BACKEND: Flask + MediaPipe FaceMesh
#  Fungsi utama:
#    - Menerima gambar dari Flutter (foto mata/wajah)
#    - Deteksi iris & pupil menggunakan MediaPipe
#    - Hitung pergerakan pupil antar frame (movement)
#    - Tentukan status mata: normal / kemungkinan_tunanetra
#    - Mengirim hasilnya kembali dalam bentuk JSON
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS     # supaya API bisa diakses dari domain lain (Flutter, Web, dll)
import cv2                     # OpenCV, untuk mengolah gambar
import numpy as np             # operasi matematika / vektor
import base64                  # decode gambar base64 (mode web)
import mediapipe as mp         # library MediaPipe untuk FaceMesh (deteksi wajah & iris)


# ============================================================
#  INISIALISASI FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app)  # mengaktifkan CORS, agar bisa diakses dari Flutter (android / web) tanpa blokir CORS


# ============================================================
#  INISIALISASI MEDIAPIPE FACEMESH
# ============================================================
# FaceMesh = model Mediapipe untuk mendeteksi:
#   - bentuk wajah (468 landmark)
#   - + iris/pupil (butuh refine_landmarks=True)
#
# Parameter:
#   max_num_faces            → hanya deteksi 1 wajah
#   refine_landmarks=True    → penting untuk deteksi iris & pupil (index 468+)
#   min_detection_confidence → kepercayaan minimal untuk deteksi wajah
#   min_tracking_confidence  → kepercayaan minimal untuk tracking landmark
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
#  INDEX LANDMARK IRIS & PUPIL (dari dokumentasi MediaPipe)
# ============================================================
# Index landmark untuk iris & pupil:
#   - Pupil kanan: center 468, ring 469-472
#   - Pupil kiri : center 473, ring 474-477
# ============================================================

RIGHT_PUPIL_CENTER = 468
RIGHT_IRIS_RING = [469, 470, 471, 472]

LEFT_PUPIL_CENTER = 473
LEFT_IRIS_RING = [474, 475, 476, 477]


# ============================================================
#  VARIABEL GLOBAL UNTUK FRAME SEBELUMNYA
# ============================================================
# Variabel ini dipakai untuk menyimpan posisi pupil
# pada frame sebelumnya, agar kita bisa menghitung
# seberapa jauh pupil berpindah (movement) antar frame.
# ============================================================

prev_left_center = None   # (x, y) pupil kiri pada frame sebelumnya
prev_right_center = None  # (x, y) pupil kanan pada frame sebelumnya


# ============================================================
#  FUNGSI: decode_image(request)
#  - Menerima request dari client
#  - Mengambil gambar dari:
#       1. multipart/form-data  (mode Android/iOS)
#       2. JSON base64          (mode Web)
#  - Mengembalikan: gambar dalam format OpenCV (numpy array BGR)
# ============================================================
def decode_image(request):
    try:
        # ----------------------------------------------------
        # 1) Jika gambar dikirim sebagai file (form-data)
        # ----------------------------------------------------
        if "image" in request.files:
            file = request.files["image"]      # ambil file dari form
            img_bytes = file.read()            # baca dalam bentuk bytes
            # decode bytes → numpy array (format BGR OpenCV)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img

        # ----------------------------------------------------
        # 2) Jika gambar dikirim dalam format JSON base64
        # ----------------------------------------------------
        if request.is_json:
            img64 = request.json.get("image")  # ambil string base64

            # format: "data:image/jpeg;base64,AAAAA..."
            if img64 and img64.startswith("data:image"):
                # pisahkan header "data:image/..." dan isi base64-nya
                _, encoded = img64.split(",", 1)
                img_bytes = base64.b64decode(encoded)  # decode base64 → bytes
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                return img
    except:
        # jika ada error (format tidak valid, dsb)
        return None

    return None


# ============================================================
#  FUNGSI: calculate_movement(prev, now)
#  - Menghitung jarak perpindahan pupil (movement)
#  - Input:
#       prev = posisi pupil frame sebelumnya (x_prev, y_prev)
#       now  = posisi pupil frame sekarang   (x_now,  y_now)
#  - Rumus:
#       movement = sqrt( (dx)^2 + (dy)^2 )  → jarak Euclidean
# ============================================================
def calculate_movement(prev, now):
    # jika belum ada data sebelumnya (frame pertama)
    if prev is None:
        return 0.0  # movement = 0 di frame pertama

    # ubah ke numpy array lalu hitung jarak Euclidean
    return float(np.linalg.norm(np.array(prev) - np.array(now)))


# ============================================================
#  FUNGSI: extract_pupil(img)
#  - Deteksi iris & pupil menggunakan MediaPipe FaceMesh
#  - Menghitung:
#       - koordinat pusat pupil kiri & kanan
#       - radius iris kiri & kanan
#       - movement pupil kiri & kanan
#  - Output: dictionary berisi data kedua mata
# ============================================================
def extract_pupil(img):
    global prev_left_center, prev_right_center  # pakai variabel global

    # ukuran gambar
    h, w = img.shape[:2]

    # MediaPipe butuh format RGB, sedangkan OpenCV → BGR
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # jalankan FaceMesh
    results = mp_face_mesh.process(rgb)

    # jika wajah tidak terdeteksi
    if not results.multi_face_landmarks:
        return None

    # ambil landmark wajah pertama (karena max_num_faces=1)
    lm = results.multi_face_landmarks[0].landmark

    # cek apakah landmark cukup sampai index 477 (478 titik)
    if len(lm) < 478:
        return None

    # --------------------------------------------------------
    #  Fungsi bantu untuk menghitung:
    #    - pusat pupil dalam pixel
    #    - radius iris (rata-rata jarak center → ring iris)
    # --------------------------------------------------------
    def calc(center_idx, ring):
        # koor. pusat pupil (dalam pixel)
        cx = int(lm[center_idx].x * w)
        cy = int(lm[center_idx].y * h)

        # titik-titik ring iris (4 titik)
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in ring]

        # hitung jarak dari pusat ke masing-masing titik ring
        d = [np.linalg.norm(np.array([cx, cy]) - np.array(p)) for p in pts]

        # radius = rata-rata jarak
        radius = float(np.mean(d))

        # kembalikan (center, radius)
        return (cx, cy), radius

    # --------------------------------------------------------
    #  Hitung pupil kanan dan kiri menggunakan fungsi calc()
    # --------------------------------------------------------
    right_center, right_radius = calc(RIGHT_PUPIL_CENTER, RIGHT_IRIS_RING)
    left_center, left_radius = calc(LEFT_PUPIL_CENTER, LEFT_IRIS_RING)

    # --------------------------------------------------------
    #  Hitung pergerakan (movement) pupil antara frame lama & baru
    # --------------------------------------------------------
    right_movement = calculate_movement(prev_right_center, right_center)
    left_movement = calculate_movement(prev_left_center, left_center)

    # simpan posisi saat ini sebagai "frame sebelumnya" untuk panggilan berikutnya
    prev_right_center = right_center
    prev_left_center = left_center

    # kembalikan data lengkap kedua mata dalam bentuk dictionary
    return {
        "right": {
            "center_x": right_center[0],
            "center_y": right_center[1],
            "radius": right_radius,
            "movement": right_movement
        },
        "left": {
            "center_x": left_center[0],
            "center_y": left_center[1],
            "radius": left_radius,
            "movement": left_movement
        }
    }


# ============================================================
#  FUNGSI: eye_status(movement_left, movement_right)
#  - Menentukan "status mata" berdasarkan pergerakan pupil
#
#  Logika sederhana:
#     - Jika gerakan kiri & kanan < threshold → kemungkinan_tunanetra
#     - Jika salah satu atau dua-duanya ≥ threshold → normal
#
#  Catatan:
#     threshold ini masih kasar (heuristik),
#     nanti bisa dituning dari data penelitian/klinik.
# ============================================================
def eye_status(movement_left, movement_right):
    threshold = 1.0  # ambang batas movement dalam pixel (bisa disesuaikan)

    # kedua mata hampir tidak bergerak (bawah threshold)
    if movement_left < threshold and movement_right < threshold:
        return "kemungkinan_tunanetra"
    else:
        # minimal satu mata bergerak cukup → dianggap normal
        return "normal"


# ============================================================
#  API ENDPOINT: /detect  (METHOD: POST)
#  - Dihubungkan dengan Flutter
#  - Alur:
#     1. Ambil gambar dari request (decode_image)
#     2. Jalankan extract_pupil untuk deteksi iris & movement
#     3. Hitung status mata dengan eye_status()
#     4. Kirim hasil JSON ke Flutter
# ============================================================
@app.route("/detect", methods=["POST"])
def detect():
    # decode gambar dari request (file / base64)
    img = decode_image(request)

    # jika gagal baca gambar
    if img is None:
        return jsonify({"error": "Gagal membaca gambar"}), 400

    # deteksi pupil + movement
    pupil = extract_pupil(img)

    # jika gagal deteksi wajah / iris
    if pupil is None:
        return jsonify({
            "status": "pupil_not_found",
            "message": "Wajah/iris tidak terdeteksi"
        })

    # tentukan status eyes berdasarkan movement pupil kiri & kanan
    status = eye_status(pupil["left"]["movement"], pupil["right"]["movement"])

    # kirim JSON lengkap:
    #  - status: normal / kemungkinan_tunanetra
    #  - pupil: data kiri & kanan (center, radius, movement)
    return jsonify({
        "status": status,
        "pupil": pupil
    })


# ============================================================
#  ENDPOINT ROOT "/" – hanya untuk mengecek server hidup
# ============================================================
@app.route("/")
def home():
    return jsonify({"status": "MediaPipe Iris Detector with Movement Tracking OK"})


# ============================================================
#  MENJALANKAN SERVER FLASK
#  host="0.0.0.0" → agar bisa diakses dari device lain di jaringan yang sama
#  port=5000      → port server
#  debug=True     → log lebih lengkap (sebaiknya False di produksi)
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
