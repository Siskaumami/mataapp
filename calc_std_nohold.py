#hitung menjalankan tanpa hold
import json
import statistics

with open("nohold_60s.json", "r", encoding="utf-8") as f:
    d = json.load(f)

data = d["data"]

print("=== TANPA HOLD ===")

# Global std dev (semua kecuali pupil_not_found)
vals = [
    (x["movement_left"] + x["movement_right"]) / 2
    for x in data
    if x.get("movement_left") is not None
    and x.get("status") != "pupil_not_found"
]

print("Global stddev:", statistics.stdev(vals))

for s in ["normal", "closed", "kemungkinan_tunanetra"]:
    v = [
        (x["movement_left"] + x["movement_right"]) / 2
        for x in data
        if x.get("status") == s
        and x.get("movement_left") is not None
    ]

    if len(v) > 1:
        print(s, "n =", len(v), "stddev =", statistics.stdev(v))
    else:
        print(s, "n =", len(v), "stddev = not enough data")