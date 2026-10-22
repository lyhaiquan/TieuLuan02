# -*- coding: utf-8 -*-
"""Duong co so khong dung mang no-ron tren DUNG cac phep chia cua lab04_data (Chuong 3).
python ch3_baselines.py -> results/baselines.json"""
import json, time
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lab04_data as L

out = {}

# ---------------------------------------------------------------- diabetes
d = L.load_diabetes()
out["diabetes"] = {}
for name, m in [("Hồi quy logistic", LogisticRegression(max_iter=1000)), ("Rừng ngẫu nhiên (200 cây)", RandomForestClassifier(200, random_state=42, n_jobs=4))]:
    t = time.perf_counter(); m.fit(d["X_train"], d["y_train"]); p = m.predict_proba(d["X_test"])[:, 1]
    out["diabetes"][name] = dict(L.binary_metrics(d["y_test"], p), seconds=time.perf_counter() - t)
    print("diabetes", name, out["diabetes"][name], flush=True)

# ---------------------------------------------------------------- housing
h = L.load_housing()


def hm(z):
    z = np.asarray(z).ravel()
    log_pred = z * h["y_sd"] + h["y_mu"]
    price = np.expm1(log_pred)
    price_true = np.expm1(h["y_test_log"])
    r_b = L.regression_metrics(price_true, price)
    r_l = L.regression_metrics(h["y_test_log"], log_pred)
    return dict(mae=r_b["mae"], rmse=r_b["rmse"], r2=r_b["r2"], mae_log=r_l["mae"], r2_log=r_l["r2"])


out["housing"] = {}
for name, m in [("Hồi quy tuyến tính", LinearRegression()), ("Rừng ngẫu nhiên (200 cây)", RandomForestRegressor(200, min_samples_leaf=3, random_state=42, n_jobs=4))]:
    t = time.perf_counter(); m.fit(h["X_train"], h["y_train"]); z = m.predict(h["X_test"])
    out["housing"][name] = dict(hm(z), seconds=time.perf_counter() - t)
    print("housing", name, out["housing"][name], flush=True)

# ---------------------------------------------------------------- comments
c = L.load_comments()
V = c["vocab_size"]


def bow(X):
    rows = np.repeat(np.arange(len(X)), X.shape[1])
    M = sp.csr_matrix((np.ones(X.size), (rows, X.ravel())), shape=(len(X), V))
    M = M[:, 2:]                      # bo cot <pad>, <unk>
    return M


Xtr, Xte = bow(c["X_train"]), bow(c["X_test"])
out["comments"] = {}
t = time.perf_counter(); m = LogisticRegression(max_iter=1000).fit(Xtr, c["y_train"]); P = m.predict_proba(Xte)
mt = L.multiclass_metrics(c["y_test"], P)
out["comments"]["Hồi quy logistic trên túi từ"] = dict(accuracy=mt["accuracy"], macro_f1=mt["macro_f1"], per_class=mt["per_class"], seconds=time.perf_counter() - t)
print("comments", out["comments"], flush=True)
json.dump(out, open("results/baselines.json", "w"), indent=1, ensure_ascii=False)
print("DONE")
