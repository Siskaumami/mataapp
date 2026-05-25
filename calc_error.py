import json
import numpy as np

def load_avg_pos(filename):
    with open(filename, "r", encoding="utf-8") as f:
        d = json.load(f)
    data = d["data"]

    xs = []
    ys = []

    for x in data:
        if x.get("any_closed") == False and x.get("status") != "pupil_not_found":
            if x.get("center_x") and x.get("center_y"):
                xs.append(x["center_x"])
                ys.append(x["center_y"])

    return np.mean(xs), np.mean(ys)

# baseline pakai 45 cm
bx, by = load_avg_pos("jarak45.json")

for jarak in ["30", "45", "60"]:
    x, y = load_avg_pos(f"jarak{jarak}.json")
    error = np.sqrt((x-bx)**2 + (y-by)**2)
    print(f"{jarak} cm error:", error)