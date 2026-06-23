import pandas as pd
from pathlib import Path

files = {
    "Tanpa Preprocessing": "hasil_tanpa_retinex.csv",
    "Retinex": "hasil_dengan_retinex.csv",
    "Multiscale CLAHE": "hasil_mclahe.csv",
    "SCI": "hasil_sci.csv",
    "IAGC": "hasil_iagc.csv",
    "Zero-DiDCE": "hasil_zerodidce.csv",
}

hasil_rekap = []

for metode, nama_file in files.items():
    path = Path(nama_file)

    if not path.exists():
        print(f"File tidak ditemukan: {nama_file}")
        continue

    df = pd.read_csv(path)

    # Membersihkan nama kolom dan isi kondisi agar aman
    df.columns = df.columns.str.strip()
    df["kondisi"] = df["kondisi"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip()

    print("\n" + "=" * 60)
    print(f"METODE: {metode}")
    print("=" * 60)

    for kondisi in ["normal", "low_light"]:
        data = df[df["kondisi"] == kondisi]

        total = len(data)

        if total == 0:
            print(f"Tidak ada data untuk kondisi: {kondisi}")
            continue

        status_normal = (data["status"] == "normal").sum()

        gerakan_minimal = (
            (data["status"] == "kemungkinan_tunanetra").sum()
            + (data["status"] == "gerakan_pupil_minimal").sum()
        )

        mata_tertutup = (data["status"] == "closed").sum()
        pupil_not_found = (data["status"] == "pupil_not_found").sum()

        success_rate = ((total - pupil_not_found) / total) * 100

        print("-" * 40)
        print(f"Kondisi pencahayaan : {kondisi}")
        print(f"Total citra          : {total}")
        print(f"Status normal        : {status_normal}")
        print(f"Gerakan pupil minimal: {gerakan_minimal}")
        print(f"Mata tertutup        : {mata_tertutup}")
        print(f"Pupil_not_found      : {pupil_not_found}")
        print(f"Success rate         : {success_rate:.2f}%")

        hasil_rekap.append({
            "Metode": metode,
            "Kondisi Pencahayaan": kondisi,
            "Total Citra": total,
            "Status Normal": status_normal,
            "Gerakan Pupil Minimal": gerakan_minimal,
            "Mata Tertutup": mata_tertutup,
            "Pupil_not_found": pupil_not_found,
            "Success Rate": f"{success_rate:.2f}%"
        })

rekap_df = pd.DataFrame(hasil_rekap)

print("\n" + "=" * 60)
print("REKAP SEMUA METODE PER KONDISI PENCAHAYAAN")
print("=" * 60)
print(rekap_df.to_string(index=False))

rekap_df.to_csv("rekap_semua_metode_per_kondisi.csv", index=False)
print("\nFile rekap tersimpan: rekap_semua_metode_per_kondisi.csv")