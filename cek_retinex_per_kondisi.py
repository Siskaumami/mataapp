import pandas as pd

df = pd.read_csv("hasil_dengan_retinex.csv")

for kondisi in ["normal", "low_light"]:
    data = df[df["kondisi"] == kondisi]

    total = len(data)
    pupil_not_found = (data["status"] == "pupil_not_found").sum()
    success_rate = ((total - pupil_not_found) / total) * 100

    status_normal = (data["status"] == "normal").sum()
    gerakan_minimal = (data["status"] == "kemungkinan_tunanetra").sum()
    mata_tertutup = (data["status"] == "closed").sum()

    print("=" * 40)
    print(f"Kondisi pencahayaan : {kondisi}")
    print(f"Total citra          : {total}")
    print(f"Status normal        : {status_normal}")
    print(f"Gerakan pupil minimal: {gerakan_minimal}")
    print(f"Mata tertutup        : {mata_tertutup}")
    print(f"Pupil_not_found      : {pupil_not_found}")
    print(f"Success rate         : {success_rate:.2f}%")