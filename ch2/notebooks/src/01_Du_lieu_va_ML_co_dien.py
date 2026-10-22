# %% [markdown]
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
