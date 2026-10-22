# -*- coding: utf-8 -*-
import subprocess, sys
from pathlib import Path
R = Path(r"C:\DATA\TieuLuan01")

# 1. ch2_run.py: bao ve phan chay chinh de import duoc tu notebook
p = R / "ch2/ch2_run.py"
s = p.read_text(encoding="utf-8")
s = s.replace("FW = sys.argv[1]\nONLY = sys.argv[2].split(\",\") if len(sys.argv) > 2 else", "FW = sys.argv[1] if len(sys.argv) > 1 else None\nONLY = sys.argv[2].split(\",\") if len(sys.argv) > 2 else")
i = s.index("fn = dict(numpy=run_numpy")
head, tail = s[:i], s[i:]
tail = "def main():\n" + "\n".join(("    " + l if l.strip() else l) for l in tail.splitlines()) + "\n\n\nif __name__ == \"__main__\":\n    main()\n"
p.write_text(head + tail, encoding="utf-8")

# 2. notebook nguon dang percent
nb2 = {}
nb2["01_Du_lieu_va_ML_co_dien"] = '''# %% [markdown]
# # Chương 2, notebook 1: dữ liệu và các mô hình học máy cổ điển
# Tiền xử lý dùng chung nằm trong `ch2_data.py` (làm sạch, mã hóa, chia tập phân tầng seed 42, chuẩn hóa chỉ fit trên Train).
# Notebook này dựng lại ba mảng `data/ch2_*.npz` rồi chạy lại các mô hình scikit-learn (mất khoảng 3 phút).

# %%
import os, sys, json
sys.path.insert(0, os.path.abspath(".."))
os.chdir(os.path.abspath(".."))
import numpy as np
import ch2_data

# %%
for name, fn in [("diabetes", ch2_data.diabetes), ("housing", ch2_data.housing), ("comments", ch2_data.comments)]:
    D = fn()
    np.savez_compressed("data/ch2_%s.npz" % name, **D)
    print(name, {k: v.shape for k, v in D.items()})

# %%
# Chạy lại các mô hình cổ điển (ghi results/ch2_sklearn.json)
exec(open("ch2_sklearn.py", encoding="utf-8").read())

# %%
res = json.load(open("results/ch2_sklearn.json", encoding="utf-8"))
for ds, models in res.items():
    print("==", ds)
    for k, v in models.items():
        print("  %-42s" % k, {a: round(b, 4) for a, b in v.items()})
'''
for fw, title, runner in [("numpy", "NumPy thuần (`dl_scratch.MLPScratch`)", "run_numpy"), ("torch", "PyTorch (`nn.Sequential`, autograd)", "run_torch"), ("keras", "Keras (`Sequential`, `fit()`)", "run_keras")]:
    k = {"numpy": "02_MLP_NumPy_scratch", "torch": "03_MLP_PyTorch", "keras": "04_MLP_Keras"}[fw]
    ker = "Kernel: **Python (TF 2.21)** (môi trường Keras)." if fw == "keras" else ""
    nb2[k] = '''# %% [markdown]
# # Chương 2: MLP bằng ''' + title + '''
# Cùng kiến trúc, khởi tạo He, SGD thuần, tốc độ học, epoch, kích thước lô với hai cài đặt còn lại (xem `CFG` trong `ch2_run.py`). ''' + ker + '''
# Notebook chạy nhanh một bộ dữ liệu một hạt giống; muốn chạy đủ 3 bộ x 3 hạt giống dùng `python ch2_run.py ''' + fw + '''`.

# %%
import os, sys, inspect
sys.path.insert(0, os.path.abspath(".."))
os.chdir(os.path.abspath(".."))
import numpy as np
import ch2_run as C

# %%
print(inspect.getsource(C.''' + runner + ''')) if "''' + fw + '''" != "numpy" else print(inspect.getsource(__import__("dl_scratch").MLPScratch))

# %%
name = "housing"                      # đổi thành "diabetes" hoặc "comments" (diabetes chạy lâu hơn)
D = dict(np.load("data/ch2_%s.npz" % name))
metrics, seconds, n_params, loss = C.''' + runner + '''(name, D, C.CFG[name], 42)
print(n_params, "tham số;", round(seconds, 1), "giây")
print({k: round(v, 4) for k, v in metrics.items()})
'''
nb2["05_So_sanh_ba_cai_dat"] = '''# %% [markdown]
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
'''
nb4 = '''# %% [markdown]
# # Chương 4, notebook 6: RNN, LSTM, GRU và MLP viết bằng NumPy thuần
# Thư viện `rnn_numpy.py`: lan truyền tiến, lan truyền ngược theo thời gian (BPTT), Adam, cắt chuẩn gradient, dừng sớm.
# Trọng số bố trí giống `torch.nn`, nên có thể nạp cùng trọng số vào PyTorch để đối chiếu từng con số.

# %%
import os, sys
sys.path.insert(0, os.path.abspath(".."))
os.chdir(os.path.abspath(".."))
import numpy as np
import rnn_numpy as R

# %% [markdown]
# ## 1. Kiểm chứng với PyTorch và sai phân hữu hạn

# %%
exec(open("verify_numpy.py", encoding="utf-8").read())

# %% [markdown]
# ## 2. Huấn luyện nhanh một hạt giống trên Churn
# Chạy đủ ba hạt giống và cả bài AMZN bằng `python run_numpy.py churn|ret|vol`.

# %%
import lab06_data as D
C = D.churn_splits(); hp = D.CHURN_HP
seq = lambda X: X[:, :, None]
data = dict(X_train=seq(C["X_train"]), y_train=C["y_train"], X_val=seq(C["X_val"]), y_val=C["y_val"])
for cell in ("RNN", "LSTM", "GRU"):
    m = R.SeqNetNP(cell, 1, hp["hidden"], 0.0, 42)
    info = R.fit(m, data, R.bce_grad, hp, 42)
    pt = R.sigm(m.forward(seq(C["X_test"]).astype(np.float64)))
    thr = D.best_threshold(C["y_val"], R.sigm(m.forward(seq(C["X_val"]).astype(np.float64))))
    print(cell, m.n_params(), "tham số", D.cls_metrics(C["y_test"], pt, thr)["roc_auc"], "ROC-AUC", info["epochs"], "epoch", info["seconds"], "giây")

# %% [markdown]
# ## 3. Kết quả đầy đủ (ba hạt giống) đã lưu

# %%
import json, glob
for f in sorted(glob.glob("results/numpy__churn__*.json")):
    d = json.load(open(f, encoding="utf-8")); print(d["model"], d["params"], d["summary"]["roc_auc"])
'''
(R / "ch2/notebooks").mkdir(exist_ok=True)
(R / "ch2/notebooks/src").mkdir(exist_ok=True)
for k, v in nb2.items():
    (R / "ch2/notebooks/src" / (k + ".py")).write_text(v, encoding="utf-8")
(R / "ch4/src" / "nb06_rnn_numpy.py").write_text(nb4, encoding="utf-8")
print("ok")
