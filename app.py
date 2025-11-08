import os
import sys
from flask import Flask, render_template, Response, request, send_file
import cv2
import mediapipe as mp
import numpy as np
import threading
from fpdf import FPDF
import io
from scipy.spatial import distance as dist

# === Konfigurasi dasar Flask ===
def get_template_path():
    try:
        base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    except Exception as e:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Error in template path: {e}")
    return os.path.join(base_path, 'templates')

app = Flask(__name__, template_folder=get_template_path())

# === Variabel global ===
blink_count = 0
closed_eye_frames = 0
detection_result = ""
camera_active = True
cap = None  # Kamera global agar tidak dibuka dua kali

# === Inisialisasi Mediapipe ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# === EAR Function (Eye Aspect Ratio) ===
def eye_aspect_ratio(landmarks, eye_indices):
    A = dist.euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = dist.euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = dist.euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    ear = (A + B) / (2.0 * C)
    return ear

# Indeks mata untuk Mediapipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# === Fungsi deteksi kedipan ===
def deteksi_kedipan():
    global blink_count, closed_eye_frames, detection_result, camera_active, cap

    ear_threshold = 0.25
    consec_frames = 2
    frame_count = 0

    if cap is None:
        cap = cv2.VideoCapture(0)

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                if ear < ear_threshold:
                    closed_eye_frames += 1
                    frame_count += 1
                    print(f"[INFO] Mata tertutup - EAR: {ear:.2f}")
                else:
                    if frame_count >= consec_frames:
                        blink_count += 1
                        print(f"[INFO] Kedipan terdeteksi! Total: {blink_count}")
                    frame_count = 0

                # Gambar mesh wajah
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

        # Tentukan status deteksi
        if blink_count > 10 and closed_eye_frames <= 50:
            detection_result = "Normal"
        elif closed_eye_frames > 100:
            detection_result = "Tunanetra"
        elif blink_count == 0 and closed_eye_frames > 10:
            detection_result = "Tunanetra"
        else:
            detection_result = "Normal"

        print(f"[STATUS] Blink: {blink_count}, ClosedFrames: {closed_eye_frames}, Result: {detection_result}")

    if cap is not None:
        cap.release()
        cap = None
    cv2.destroyAllWindows()

# === ROUTE Flask ===
@app.route('/')
def index():
    global camera_active
    camera_active = True
    detection_thread = threading.Thread(target=deteksi_kedipan)
    detection_thread.daemon = True
    detection_thread.start()
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    nama = request.form['nama']
    email = request.form['email']
    tanggal_lahir = request.form['tanggal_lahir']
    alamat = request.form['alamat']
    pekerjaan = request.form['pekerjaan']
    hobi = request.form['hobi']
    jurusan = request.form['jurusan']
    jalur = request.form['jalur']
    nama_orang_tua = request.form['nama_orang_tua']
    alamat_orang_tua = request.form['alamat_orang_tua']
    phone_orang_tua = request.form['phone_orang_tua']
    pesan = request.form['pesan']

    global camera_active, detection_result
    camera_active = False  # Matikan kamera setelah form dikirim

    return render_template(
        'hasil.html',
        nama=nama,
        email=email,
        tanggal_lahir=tanggal_lahir,
        alamat=alamat,
        pekerjaan=pekerjaan,
        hobi=hobi,
        jurusan=jurusan,
        jalur=jalur,
        nama_orang_tua=nama_orang_tua,
        alamat_orang_tua=alamat_orang_tua,
        phone_orang_tua=phone_orang_tua,
        pesan=pesan,
        hasil_deteksi=detection_result
    )

# === Streaming video ke browser ===
def gen():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if cap is not None:
        cap.release()
        cap = None

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === ROUTE Download PDF ===
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    nama = request.form.get('nama', '')
    email = request.form.get('email', '')
    tanggal_lahir = request.form.get('tanggal_lahir', '')
    alamat = request.form.get('alamat', '')
    pekerjaan = request.form.get('pekerjaan', '')
    hobi = request.form.get('hobi', '')
    jurusan = request.form.get('jurusan', '')
    jalur = request.form.get('jalur', '')
    nama_orang_tua = request.form.get('nama_orang_tua', '')
    alamat_orang_tua = request.form.get('alamat_orang_tua', '')
    phone_orang_tua = request.form.get('phone_orang_tua', '')
    pesan = request.form.get('pesan', '')
    hasil_deteksi = request.form.get('hasil_deteksi', '')

    # Buat PDF dengan FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "FORMULIR PENDAFTARAN MAHASISWA", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    fields = [
        ("Nama", nama),
        ("Email", email),
        ("Tanggal Lahir", tanggal_lahir),
        ("Alamat", alamat),
        ("Pekerjaan", pekerjaan),
        ("Hobi", hobi),
        ("Jurusan", jurusan),
        ("Jalur", jalur),
        ("Nama Orang Tua", nama_orang_tua),
        ("Alamat Orang Tua", alamat_orang_tua),
        ("No. Telepon Orang Tua", phone_orang_tua),
        ("Pesan", pesan),
        ("Hasil Deteksi Kedipan", hasil_deteksi)
    ]
    for label, value in fields:
        pdf.cell(60, 10, f"{label}:", 0, 0)
        pdf.multi_cell(0, 10, value)

    pdf_output = pdf.output(dest='S').encode('latin1')
    return send_file(
        io.BytesIO(pdf_output),
        as_attachment=True,
        download_name=f"hasil_deteksi_{nama.replace(' ', '_')}.pdf",
        mimetype='application/pdf'
    )

# === Jalankan Flask ===
if __name__ == "__main__":
    app.run(debug=False, threaded=False)
