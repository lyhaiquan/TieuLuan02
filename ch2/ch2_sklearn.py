# -*- coding: utf-8 -*-
"""Cac mo hinh ML co dien tren DUNG cung X, y (chia tap, chuan hoa) voi ba cai dat MLP.
python ch2_sklearn.py  ->  results/ch2_sklearn.json"""
import json, time
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

res = {}
rng = np.random.RandomState(42)


def sub(X, y, k):
    i = rng.choice(len(X), min(k, len(X)), replace=False)
    return X[i], y[i]


def bin_m(y, p):
    yh = (p >= .5).astype(int); y = y.ravel()
    tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(accuracy=float((yh == y).mean()), precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12), roc_auc=float(roc_auc_score(y, p)))


def timed(fn):
    t = time.perf_counter(); r = fn(); return r, time.perf_counter() - t


# ---------------------------------------------------------------- diabetes
D = dict(np.load("data/ch2_diabetes.npz")); Xtr, Xte, ytr, yte = D["X_tr"], D["X_te"], D["y_tr"].ravel(), D["y_te"].ravel()
out = {}
for name, mk, sub_n in [("Hồi quy logistic", lambda: LogisticRegression(max_iter=1000), None), ("KNN (k=15)", lambda: KNeighborsClassifier(15, n_jobs=4), None),
                        ("SVM nhân RBF (20.000 mẫu huấn luyện)", lambda: SVC(C=1.0, probability=False), 20000),
                        ("Cây quyết định (sâu 6)", lambda: DecisionTreeClassifier(max_depth=6, random_state=42), None),
                        ("Rừng ngẫu nhiên (200 cây)", lambda: RandomForestClassifier(200, random_state=42, n_jobs=4), None)]:
    X1, y1 = (Xtr, ytr) if sub_n is None else sub(Xtr, ytr, sub_n)
    def run():
        m = mk().fit(X1, y1)
        if hasattr(m, "predict_proba"): return m.predict_proba(Xte)[:, 1]
        return 1 / (1 + np.exp(-m.decision_function(Xte)))
    p, s = timed(run); out[name] = dict(bin_m(yte, p), seconds=s); print("diabetes", name, out[name], flush=True)
res["diabetes"] = out

# ---------------------------------------------------------------- housing
H = dict(np.load("data/ch2_housing.npz")); Xtr, Xte, ytr, yte = H["X_tr"], H["X_te"], H["y_tr"].ravel(), H["y_te"].ravel()
ysd, ymu, pt = float(H["y_sd"]), float(H["y_mu"]), H["price_te"]
out = {}
def reg_m(z):
    price = np.exp(z * ysd + ymu)
    return dict(r2_log=float(1 - ((z - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()), r2_price=float(1 - ((price - pt) ** 2).sum() / ((pt - pt.mean()) ** 2).sum()),
                mae=float(np.abs(price - pt).mean()), rmse=float(np.sqrt(((price - pt) ** 2).mean())), mape=float((np.abs(price - pt) / pt).mean()))
for name, mk, sub_n in [("Hồi quy tuyến tính", LinearRegression, None), ("KNN (k=15)", lambda: KNeighborsRegressor(15, n_jobs=4), None),
                        ("SVR nhân RBF (20.000 mẫu huấn luyện)", lambda: SVR(C=1.0), 20000), ("Cây quyết định (sâu 8)", lambda: DecisionTreeRegressor(max_depth=8, random_state=42), None),
                        ("Rừng ngẫu nhiên (200 cây)", lambda: RandomForestRegressor(200, min_samples_leaf=3, random_state=42, n_jobs=4), None)]:
    X1, y1 = (Xtr, ytr) if sub_n is None else sub(Xtr, ytr, sub_n)
    z, s = timed(lambda: mk().fit(X1, y1).predict(Xte)); out[name] = dict(reg_m(z), seconds=s); print("housing", name, out[name], flush=True)
res["housing"] = out

# ---------------------------------------------------------------- comments
Cm = dict(np.load("data/ch2_comments.npz")); Xtr, Xte, ytr, yte = Cm["X_tr"].astype(np.float32), Cm["X_te"].astype(np.float32), Cm["y_tr"], Cm["y_te"]
out = {}
def mc_m(P):
    yh = P.argmax(1); f = []
    for k in range(3):
        tp = ((yh == k) & (yte == k)).sum(); fp = ((yh == k) & (yte != k)).sum(); fn = ((yh != k) & (yte == k)).sum()
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1); f.append(2 * p_ * r_ / max(p_ + r_, 1e-12))
    return dict(accuracy=float((yh == yte).mean()), macro_f1=float(np.mean(f)), f1_neg=float(f[0]), f1_neu=float(f[1]), f1_pos=float(f[2]))
def sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
for name, mk in [("Hồi quy logistic đa lớp", lambda: LogisticRegression(max_iter=1000)), ("Naive Bayes đa thức", lambda: MultinomialNB()),
                 ("SVM tuyến tính", lambda: LinearSVC(C=0.5)), ("Rừng ngẫu nhiên (200 cây)", lambda: RandomForestClassifier(200, random_state=42, n_jobs=4))]:
    def run():
        m = mk().fit(Xtr, ytr)
        return m.predict_proba(Xte) if hasattr(m, "predict_proba") else sm(m.decision_function(Xte))
    P, s = timed(run); out[name] = dict(mc_m(P), seconds=s); print("comments", name, out[name], flush=True)
res["comments"] = out
json.dump(res, open("results/ch2_sklearn.json", "w"), indent=1, ensure_ascii=False)
print("DONE")
