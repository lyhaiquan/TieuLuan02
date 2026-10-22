# %% [markdown]
# # Chương 4, notebook 6: RNN, LSTM, GRU và MLP viết bằng NumPy thuần
# Thư viện `rnn_numpy.py`: lan truyền tiến, lan truyền ngược theo thời gian (BPTT), Adam, cắt chuẩn gradient, dừng sớm.
# Trọng số bố trí giống `torch.nn`, nên có thể nạp cùng trọng số vào PyTorch để đối chiếu từng con số.

# %%
import os, sys
while not os.path.exists("rnn_numpy.py") and os.path.dirname(os.getcwd()) != os.getcwd():
    os.chdir("..")          # di len den thu muc co rnn_numpy.py
sys.path.insert(0, os.getcwd())
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
