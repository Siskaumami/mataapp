import os
import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# KONFIGURASI
# ==========================================
MODEL_PATH = "model_wajah.tflite"
TEST_DIR = "dataset_ujian_cahaya" # Folder isi foto target dengan BANYAK VARIASI CAHAYA

CLASS_NAMES = ["Orang_A", "Orang_B", "Unknown"] 
TARGET_SIZE = (128, 128)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kategori Pencahayaan
kategori_cahaya = {
    "Terang_Ideal": {"total": 0, "benar": 0},
    "Remang_Sebagian": {"total": 0, "benar": 0},
    "Gelap_Ekstrem": {"total": 0, "benar": 0}
}

def hitung_kecerahan(img):
    """Menghitung rata-rata nilai kegelapan/keterangan piksel (0 - 255)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2]) # Mengambil channel V (Value)

print("Memulai Pengujian Faktor Pencahayaan...")

for class_index, class_name in enumerate(CLASS_NAMES):
    folder_path = os.path.join(TEST_DIR, class_name)
    if not os.path.exists(folder_path): continue

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        # 1. Analisis Kecerahan Gambar
        brightness = hitung_kecerahan(img)
        if brightness > 100:
            kategori = "Terang_Ideal"
        elif brightness > 40:
            kategori = "Remang_Sebagian"
        else:
            kategori = "Gelap_Ekstrem"
            
        kategori_cahaya[kategori]["total"] += 1

        # 2. AI Menebak
        img_resized = cv2.resize(img, TARGET_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_idx = np.argmax(output_data[0])
        
        # 3. Hitung Benar/Salah
        if predicted_idx == class_index:
            kategori_cahaya[kategori]["benar"] += 1

# ==========================================
# RAPOR PENCAHAYAAN
# ==========================================
print("\n" + "="*50)
print("HASIL PENGUJIAN BERDASARKAN FAKTOR CAHAYA")
print("="*50)

for kat, data in kategori_cahaya.items():
    if data["total"] > 0:
        akurasi = (data["benar"] / data["total"]) * 100
        print(f"Kondisi {kat:<15} : Akurasi {akurasi:05.2f}% (Benar {data['benar']} dari {data['total']} foto)")
    else:
        print(f"Kondisi {kat:<15} : Belum ada data foto")