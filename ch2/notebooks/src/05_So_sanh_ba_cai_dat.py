# %% [markdown]
# # Chương 2, notebook 5: so sánh ba cài đặt
# Đọc `results/ch2_numpy.json`, `ch2_torch.json`, `ch2_keras.json` (ba hạt giống) và `ch2_sklearn.json`.

# %%
import json, os
os.chdir(os.path.abspath(".."))
import pandas as pd
R = {fw: json.load(open("results/ch2_%s.json" % fw, encoding="utf-8")) for fw in ("numpy", "torch", "keras")}
for ds in ("diabetes", "housing", "comments"):
    rows = []
    for fw in R:
        m, s = R[fw][ds]["mean"], R[fw][ds]["std"]
        rows.append({"framework": fw, **{k: "%.4f ± %.4f" % (m[k], s[k]) for k in m}})
    print("==", ds)
    print(pd.DataFrame(rows).to_string(index=False))
