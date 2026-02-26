import json

LOG_FILE = "blink_log.json"
BLINK_SECONDS = list(range(5, 61, 5))  # 5,10,...,60
WINDOW_HALF_SEC = 0.5  # ±0.5 detik

def in_any_window(t_rel_sec: float) -> bool:
    for s in BLINK_SECONDS:
        if (s - WINDOW_HALF_SEC) <= t_rel_sec <= (s + WINDOW_HALF_SEC):
            return True
    return False

with open(LOG_FILE, "r", encoding="utf-8") as f:
    j = json.load(f)

data = j["data"]
if not data:
    raise SystemExit("Log kosong. Pastikan sudah ada request /detect.")

t0 = data[0]["t_ms"]

TP = FP = FN = TN = 0

for row in data:
    status = row.get("status")
    t_rel_sec = (row["t_ms"] - t0) / 1000.0

    gt_closed = in_any_window(t_rel_sec)
    pred_closed = (status == "closed")  # untuk evaluasi kedipan

    if pred_closed and gt_closed:
        TP += 1
    elif pred_closed and not gt_closed:
        FP += 1
    elif not pred_closed and gt_closed:
        FN += 1
    else:
        TN += 1

precision = TP / (TP + FP) if (TP + FP) else 0.0
recall    = TP / (TP + FN) if (TP + FN) else 0.0
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

print("=== Confusion Matrix (Blink) ===")
print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print("=== Metrics ===")
print(f"Precision={precision*100:.2f}%")
print(f"Recall   ={recall*100:.2f}%")
print(f"F1-score ={f1*100:.2f}%")
print(f"Total frames={len(data)}")
print(f"Duration(sec)≈{(data[-1]['t_ms']-t0)/1000.0:.2f}")
