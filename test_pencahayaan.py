import os
import csv
import json
import time
from pathlib import Path

import requests


# =========================
# KONFIGURASI
# =========================

API_URL = "http://127.0.0.1:5000"

# Folder dataset kamu
DATASET_DIR = Path("dataset_ujian_cahaya")

# Sesuaikan dengan nama folder kamu
KONDISI_FOLDER = {
    "normal": "normal",
    "low_light": "low_light"
}

# Ganti nama ini sesuai mode uji
# Kalau app.py RETINEX_ENABLED = True  -> pakai "dengan_retinex"
# Kalau app.py RETINEX_ENABLED = False -> pakai "tanpa_retinex"
NAMA_UJI =  "mclahe"

OUTPUT_CSV = f"hasil_{NAMA_UJI}.csv"
OUTPUT_STATS = f"stats_{NAMA_UJI}.json"
OUTPUT_LOG = f"log_{NAMA_UJI}.json"

EXT_GAMBAR = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# =========================
# FUNGSI BANTU
# =========================

def cek_backend():
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        print("Backend aktif:", response.json())
        return True
    except Exception as e:
        print("Backend belum aktif atau tidak bisa diakses.")
        print("Pastikan kamu sudah menjalankan: python app.py")
        print("Error:", e)
        return False


def reset_backend():
    try:
        response = requests.post(f"{API_URL}/reset_experiment", timeout=5)
        print("Reset eksperimen:", response.json())
    except Exception as e:
        print("Gagal reset eksperimen:", e)


def ambil_file_gambar(folder_path):
    if not folder_path.exists():
        print(f"Folder tidak ditemukan: {folder_path}")
        return []

    files = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in EXT_GAMBAR:
            files.append(file)

    return sorted(files)


def kirim_gambar(image_path):
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{API_URL}/detect",
                files={"image": f},
                timeout=20
            )

        return response.json()

    except Exception as e:
        return {
            "status": "request_error",
            "error": str(e)
        }


def safe_get(data, keys, default=None):
    """
    Mengambil data nested dictionary dengan aman.
    Contoh:
    safe_get(data, ["pupil", "left", "center_x"])
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


# =========================
# PROGRAM UTAMA
# =========================

def main():
    print("=" * 60)
    print("MEMULAI PENGUJIAN DATASET PENCAHAYAAN")
    print("=" * 60)

    if not cek_backend():
        return

    reset_backend()

    rows = []

    total_gambar = 0

    for kondisi, nama_folder in KONDISI_FOLDER.items():
        folder_path = DATASET_DIR / nama_folder
        gambar_list = ambil_file_gambar(folder_path)

        print()
        print(f"Kondisi: {kondisi}")
        print(f"Folder : {folder_path}")
        print(f"Jumlah gambar ditemukan: {len(gambar_list)}")

        if len(gambar_list) == 0:
            print("Belum ada data foto.")
            continue

        for idx, image_path in enumerate(gambar_list, start=1):
            print(f"[{kondisi}] Proses {idx}/{len(gambar_list)}: {image_path.name}")

            hasil = kirim_gambar(image_path)

            row = {
                "nama_uji": NAMA_UJI,
                "kondisi": kondisi,
                "filename": image_path.name,
                "status": hasil.get("status"),
                "left_eye_state": safe_get(hasil, ["eye_state", "left"]),
                "right_eye_state": safe_get(hasil, ["eye_state", "right"]),

                "ear_left": safe_get(hasil, ["ear", "left"]),
                "ear_right": safe_get(hasil, ["ear", "right"]),

                "left_center_x": safe_get(hasil, ["pupil", "left", "center_x"]),
                "left_center_y": safe_get(hasil, ["pupil", "left", "center_y"]),
                "right_center_x": safe_get(hasil, ["pupil", "right", "center_x"]),
                "right_center_y": safe_get(hasil, ["pupil", "right", "center_y"]),

                "left_movement_norm": safe_get(hasil, ["pupil", "left", "movement_norm"]),
                "right_movement_norm": safe_get(hasil, ["pupil", "right", "movement_norm"]),

                "left_radius": safe_get(hasil, ["pupil", "left", "radius"]),
                "right_radius": safe_get(hasil, ["pupil", "right", "radius"]),

                "left_eye_width": safe_get(hasil, ["geom", "left_eye_width"]),
                "right_eye_width": safe_get(hasil, ["geom", "right_eye_width"]),

                "left_rel_norm_x": safe_get(hasil, ["geom", "left_rel_norm"], [None, None])[0]
                if safe_get(hasil, ["geom", "left_rel_norm"]) else None,

                "left_rel_norm_y": safe_get(hasil, ["geom", "left_rel_norm"], [None, None])[1]
                if safe_get(hasil, ["geom", "left_rel_norm"]) else None,

                "right_rel_norm_x": safe_get(hasil, ["geom", "right_rel_norm"], [None, None])[0]
                if safe_get(hasil, ["geom", "right_rel_norm"]) else None,

                "right_rel_norm_y": safe_get(hasil, ["geom", "right_rel_norm"], [None, None])[1]
                if safe_get(hasil, ["geom", "right_rel_norm"]) else None,

                "error": hasil.get("error")
            }

            rows.append(row)
            total_gambar += 1

            time.sleep(0.1)

    print()
    print("=" * 60)
    print("MENYIMPAN HASIL")
    print("=" * 60)

    if rows:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV tersimpan: {OUTPUT_CSV}")
    else:
        print("Tidak ada hasil yang disimpan karena tidak ada gambar terbaca.")

    try:
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Stats tersimpan: {OUTPUT_STATS}")
    except Exception as e:
        print("Gagal mengambil stats:", e)

    try:
        log = requests.get(f"{API_URL}/export_blink_log", timeout=5).json()
        with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f"Log tersimpan: {OUTPUT_LOG}")
    except Exception as e:
        print("Gagal mengambil log:", e)

    print()
    print("=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    print(f"Nama uji      : {NAMA_UJI}")
    print(f"Total gambar  : {total_gambar}")

    if rows:
        counter = {}
        for row in rows:
            status = row["status"]
            counter[status] = counter.get(status, 0) + 1

        print("Status deteksi:")
        for status, jumlah in counter.items():
            print(f"- {status}: {jumlah}")

    print()
    print("Pengujian selesai.")


if __name__ == "__main__":
    main()