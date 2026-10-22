# %% [markdown]
# # Bài 06 - Tổng hợp: PyTorch và Keras, RNN và các đối chứng
#
# Notebook này không huấn luyện gì. Nó đọc các file `results/*.json` do notebook 03
# (PyTorch) và 04 (Keras) ghi ra, dựng bảng so sánh và trả lời ba câu hỏi:
#
# 1. Trên cùng dữ liệu, cùng siêu tham số, hai framework có cho cùng kết quả không?
# 2. RNN, LSTM, GRU có hơn các đối chứng không cần hồi quy không?
# 3. Chênh lệch giữa các mô hình có lớn hơn chênh lệch do hạt giống ngẫu nhiên không?

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lab06_data as D

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})
pd.set_option("display.width", 150)
R = D.load_results()
B = R["baselines"]
FW = ("torch", "keras")
FW_NAME = {"torch": "PyTorch", "keras": "Keras"}


def res(fw, task, model):
    return R["%s__%s__%s" % (fw, task, model)]


def ms(fw, task, model, key, fmt="%.4f"):
    m, s = res(fw, task, model)["summary"][key]
    return (fmt + " ± " + fmt) % (m, s)


def mean_time(fw, task, model):
    return np.mean([r["seconds"] for r in res(fw, task, model)["runs"]])


print(len(R), "file ket qua:", ", ".join(sorted(R)))

# %% [markdown]
# ## 1. Churn

# %%
rows = [dict(mo_hinh="LogReg (sklearn)", **{"AUC " + FW_NAME[f]: "%.4f" % B["churn_logreg"]["roc_auc"] for f in FW},
             **{"F1 " + FW_NAME[f]: "%.4f" % B["churn_logreg"]["f1"] for f in FW})]
for m in ("MLP", "RNN", "LSTM", "GRU"):
    row = dict(mo_hinh=m)
    for f in FW:
        row["AUC " + FW_NAME[f]] = ms(f, "churn", m, "roc_auc")
    for f in FW:
        row["F1 " + FW_NAME[f]] = ms(f, "churn", m, "f1")
    for f in FW:
        row["tham so " + FW_NAME[f]] = res(f, "churn", m)["params"]
    for f in FW:
        row["giay/lan " + FW_NAME[f]] = "%.1f" % mean_time(f, "churn", m)
    rows.append(row)
churn_cmp = pd.DataFrame(rows).set_index("mo_hinh")
churn_cmp

# %%
pt = np.array(res("torch", "churn", "LSTM")["p_test"])
pk = np.array(res("keras", "churn", "LSTM")["p_test"])
C = D.churn_splits()
thr_t = res("torch", "churn", "LSTM")["runs"][0]["metrics"]["threshold"]
thr_k = res("keras", "churn", "LSTM")["runs"][0]["metrics"]["threshold"]
agree = np.mean((pt >= thr_t) == (pk >= thr_k))
print("LSTM hat giong 42: tuong quan xac suat giua hai framework %.4f" % np.corrcoef(pt, pk)[0, 1])
print("ty le khach duoc hai framework xep cung lop: %.4f" % agree)
for m in ("MLP", "RNN", "LSTM", "GRU"):
    a_t = [r["metrics"]["roc_auc"] for r in res("torch", "churn", m)["runs"]]
    a_k = [r["metrics"]["roc_auc"] for r in res("keras", "churn", m)["runs"]]
    print("%-4s chenh AUC Keras - PyTorch: %+.4f | DLC theo hat giong: PyTorch %.4f, Keras %.4f"
          % (m, np.mean(a_k) - np.mean(a_t), np.std(a_t), np.std(a_k)))

# %% [markdown]
# ## 2. AMZN

# %%
rows = []
for name, m in B["ret"].items():
    rows.append(dict(mo_hinh=name, **{"RMSE " + FW_NAME[f]: "%.5f" % m["rmse"] for f in FW},
                     **{"dir_acc " + FW_NAME[f]: ("-" if np.isnan(m["dir_acc"]) else "%.4f" % m["dir_acc"]) for f in FW}))
for c in ("RNN", "LSTM", "GRU"):
    row = dict(mo_hinh=c)
    for f in FW:
        row["RMSE " + FW_NAME[f]] = ms(f, "ret", c, "rmse", "%.5f")
    for f in FW:
        row["dir_acc " + FW_NAME[f]] = ms(f, "ret", c, "dir_acc")
    rows.append(row)
ret_cmp = pd.DataFrame(rows).set_index("mo_hinh")
print("Bai toan 1 - loi suat phien ke tiep. Ty le phien tang trong test: %.4f" % B["up_share_test"])
ret_cmp

# %%
rows = []
for name, m in B["vol"].items():
    rows.append(dict(mo_hinh=name, **{"R2 " + FW_NAME[f]: "%.4f" % m["r2"] for f in FW},
                     **{"RMSE " + FW_NAME[f]: "%.5f" % m["rmse"] for f in FW}))
for c in ("RNN", "LSTM", "GRU"):
    row = dict(mo_hinh=c)
    for f in FW:
        row["R2 " + FW_NAME[f]] = ms(f, "vol", c, "r2")
    for f in FW:
        row["RMSE " + FW_NAME[f]] = ms(f, "vol", c, "rmse", "%.5f")
    for f in FW:
        row["giay/lan " + FW_NAME[f]] = "%.1f" % mean_time(f, "vol", c)
    rows.append(row)
vol_cmp = pd.DataFrame(rows).set_index("mo_hinh")
print("Bai toan 2 - bien dong 5 phien toi")
vol_cmp

# %% [markdown]
# ## 3. Hình tổng hợp

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
models = ("MLP", "RNN", "LSTM", "GRU")
x = np.arange(len(models))
for j, f in enumerate(FW):
    mean = [res(f, "churn", m)["summary"]["roc_auc"][0] for m in models]
    std = [res(f, "churn", m)["summary"]["roc_auc"][1] for m in models]
    axes[0].bar(x + (j - 0.5) * 0.38, mean, 0.38, yerr=std, capsize=3,
                label=FW_NAME[f], color=["#DD8452", "#8172B2"][j])
axes[0].axhline(B["churn_logreg"]["roc_auc"], c="#937860", ls="--", label="LogReg")
axes[0].set_xticks(x, models)
axes[0].set_ylim(0.75, 0.88)
axes[0].set_title("Churn: ROC-AUC trên test")
axes[0].legend(fontsize=8)

cells = ("RNN", "LSTM", "GRU")
x = np.arange(len(cells))
for j, f in enumerate(FW):
    mean = [res(f, "vol", c)["summary"]["r2"][0] for c in cells]
    std = [res(f, "vol", c)["summary"]["r2"][1] for c in cells]
    axes[1].bar(x + (j - 0.5) * 0.38, mean, 0.38, yerr=std, capsize=3,
                label=FW_NAME[f], color=["#DD8452", "#8172B2"][j])
ridge_name = [k for k in B["vol"] if k.startswith("Ridge")][0]
axes[1].axhline(B["vol"][ridge_name]["r2"], c="#55A868", ls="--", label="Ridge")
axes[1].axhline(0, c="k", lw=0.8)
axes[1].set_xticks(x, cells)
axes[1].set_title("AMZN biến động 5 phiên: R² trên test")
axes[1].legend(fontsize=8)

tasks = [("churn", m) for m in models] + [("vol", c) for c in cells]
labels = ["churn\n" + m for m in models] + ["vol\n" + c for c in cells]
x = np.arange(len(tasks))
for j, f in enumerate(FW):
    axes[2].bar(x + (j - 0.5) * 0.38, [mean_time(f, t, m) for t, m in tasks], 0.38,
                label=FW_NAME[f], color=["#DD8452", "#8172B2"][j])
axes[2].set_xticks(x, labels, fontsize=8)
axes[2].set_ylabel("giây cho một lần huấn luyện")
axes[2].set_title("Thời gian huấn luyện trên CPU (8 luồng)")
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(D.RESULTS / "fig_compare.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Ghi bảng tổng hợp

# %%
rows = []
for f in FW:
    for task, keys in (("churn", ("roc_auc", "pr_auc", "f1")), ("ret", ("rmse", "dir_acc")),
                       ("vol", ("rmse", "r2"))):
        for m in (("MLP", "RNN", "LSTM", "GRU") if task == "churn" else cells):
            r = res(f, task, m)
            rows.append(dict(framework=FW_NAME[f], task=task, model=m, params=r["params"],
                             seconds=round(mean_time(f, task, m), 1),
                             epochs=round(np.mean([x["epochs"] for x in r["runs"]]), 1),
                             **{k: round(r["summary"][k][0], 5) for k in keys},
                             **{k + "_std": round(r["summary"][k][1], 5) for k in keys}))
summary = pd.DataFrame(rows)
summary.to_csv(D.RESULTS / "summary.csv", index=False)
with open(D.RESULTS / "summary.md", "w", encoding="utf-8") as fh:
    fh.write("# Bai 06 - bang tong hop\n\n")
    for title, df in (("Churn", churn_cmp), ("AMZN loi suat", ret_cmp), ("AMZN bien dong", vol_cmp)):
        fh.write("## %s\n\n%s\n\n" % (title, df.to_markdown()))
print(summary.to_string(index=False))
