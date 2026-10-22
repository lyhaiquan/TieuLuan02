# -*- coding: utf-8 -*-
"""Ve cac hinh do tieu luan tu sinh (khong co san trong cac bai thuc hanh)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"numpy": "#C4453C", "torch": "#2B6CB0", "keras": "#2F855A"}
LAB = {"numpy": "NumPy (scratch)", "torch": "PyTorch", "keras": "Keras"}
R = {fw: json.loads((ROOT / ("ch2/results/ch2_%s.json" % fw)).read_text(encoding="utf-8")) for fw in COL}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for a, (ds, ttl, yl) in zip(ax, [("diabetes", "A. Diabetes (BCE, full-batch)", "BCE"), ("housing", "B. House Price (MSE/2, lô 1.024)", "MSE/2"), ("comments", "C. Comments (cross-entropy, lô 256)", "cross-entropy")]):
    for fw in COL:
        a.plot(R[fw][ds]["runs"][0]["loss"], c=COL[fw], lw=1.6, label=LAB[fw], ls={"numpy": "-", "torch": "--", "keras": ":"}[fw])
    a.set_title(ttl); a.set_xlabel("epoch"); a.set_ylabel(yl)
    if ds == "diabetes":
        a.set_yscale("log")
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(ROOT / "figs/c2_curves.png", bbox_inches="tight")
print("ok")
