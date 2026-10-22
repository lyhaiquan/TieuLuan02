# %% [markdown]
# # Chương 2: MLP bằng PyTorch (`nn.Sequential`, autograd)
# Cùng kiến trúc, khởi tạo He, SGD thuần, tốc độ học, epoch, kích thước lô với hai cài đặt còn lại (xem `CFG` trong `ch2_run.py`). 
# Notebook chạy nhanh một bộ dữ liệu một hạt giống; muốn chạy đủ 3 bộ x 3 hạt giống dùng `python ch2_run.py torch`.

# %%
import os, sys, inspect
sys.path.insert(0, os.path.abspath(".."))
os.chdir(os.path.abspath(".."))
import numpy as np
import ch2_run as C

# %%
print(inspect.getsource(C.run_torch)) if "torch" != "numpy" else print(inspect.getsource(__import__("dl_scratch").MLPScratch))

# %%
name = "housing"                      # đổi thành "diabetes" hoặc "comments" (diabetes chạy lâu hơn)
D = dict(np.load("data/ch2_%s.npz" % name))
metrics, seconds, n_params, loss = C.run_torch(name, D, C.CFG[name], 42)
print(n_params, "tham số;", round(seconds, 1), "giây")
print({k: round(v, 4) for k, v in metrics.items()})
