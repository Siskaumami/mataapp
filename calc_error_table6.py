import json
import math
import sys

# pakai argumen file kalau ada
fname = sys.argv[1] if len(sys.argv) > 1 else "jarak45_60s.json"

d = json.load(open(fname, "r", encoding="utf-8"))
data = d.get("data", [])

# ambil data valid (bukan pupil_not_found)
rows = [x for x in data if x.get("status") != "pupil_not_found" and x.get("geom")]

if len(rows) < 20:
    print("Data terlalu sedikit atau geom tidak ada.")
    sys.exit()

# =========================
# BASELINE (10 frame awal)
# =========================
base = rows[:10]

def avg_base(key):
    xs = []
    ys = []
    for r in base:
        L = r["geom"][f"left_{key}"]
        R = r["geom"][f"right_{key}"]
        x = (L[0] + R[0]) / 2
        y = (L[1] + R[1]) / 2
        xs.append(x)
        ys.append(y)
    return (sum(xs)/len(xs), sum(ys)/len(ys))

base_px = avg_base("rel_px")
base_norm = avg_base("rel_norm")

# =========================
# ERROR FUNCTION
# =========================
def calc_error(key, base_xy):
    errs = []
    for r in rows:
        L = r["geom"][f"left_{key}"]
        R = r["geom"][f"right_{key}"]
        x = (L[0] + R[0]) / 2
        y = (L[1] + R[1]) / 2

        dx = x - base_xy[0]
        dy = y - base_xy[1]
        dist = math.sqrt(dx*dx + dy*dy)
        errs.append(dist)

    mae = sum(errs) / len(errs)
    rmse = math.sqrt(sum(e*e for e in errs) / len(errs))
    return mae, rmse, len(errs)

mae_px, rmse_px, n1 = calc_error("rel_px", base_px)
mae_norm, rmse_norm, n2 = calc_error("rel_norm", base_norm)

print("FILE:", fname)
print("n =", n1)
print("Tanpa Normalisasi (px):")
print("  MAE  =", mae_px)
print("  RMSE =", rmse_px)
print("Dengan Normalisasi:")
print("  MAE  =", mae_norm)
print("  RMSE =", rmse_norm)