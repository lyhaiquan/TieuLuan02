# -*- coding: utf-8 -*-
"""Chuong 1 - tai hien lich su AI bang thuc nghiem tren 3 bo du lieu:
Iris (1936), MNIST (1998), Diabetes (Kaggle, thoi ky hoc may hien dai).
Chay: python ch1_history.py xor|iris|mnist|diabetes"""
import json, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
B, R, G, O = "#2B6CB0", "#C4453C", "#2F855A", "#B7791F"
FIG = "../figs/"
which = sys.argv[1]
path = "results_ch1_%s.json" % which
out = {}


def sg(z):
    return 1 / (1 + np.exp(-z))


if which == "xor":
    class Perceptron:                        # Rosenblatt 1958: w <- w + eta (y - yhat) x
        def __init__(s, n, lr=.1, seed=0):
            s.w = np.random.default_rng(seed).normal(0, .1, n)
            s.b = 0.
            s.lr = lr

        def predict(s, X):
            return (X @ s.w + s.b > 0).astype(int)

        def fit(s, X, y, epochs=200):
            hist = []
            for _ in range(epochs):
                err = 0
                for xi, yi in zip(X, y):
                    d = yi - (xi @ s.w + s.b > 0)
                    s.w += s.lr * d * xi
                    s.b += s.lr * d
                    err += int(d != 0)
                hist.append(err)
                if err == 0:
                    break
            return hist

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    res = {}
    for nm, yd in {"AND": np.array([0, 0, 0, 1]), "OR": np.array([0, 1, 1, 1]), "XOR": y}.items():
        p = Perceptron(2)
        h = p.fit(X, yd)
        res[nm] = dict(acc=float((p.predict(X) == yd).mean()), epochs=len(h), converged=bool(h[-1] == 0))
    rng = np.random.default_rng(1)
    W1 = rng.normal(0, 1, (2, 4)); b1 = np.zeros(4); W2 = rng.normal(0, 1, (4, 1)); b2 = np.zeros(1)
    for ep in range(5000):
        H = np.tanh(X @ W1 + b1)
        P = sg(H @ W2 + b2)
        d2 = (P - y[:, None]) / 4
        dH = (d2 @ W2.T) * (1 - H ** 2)
        W2 -= H.T @ d2; b2 -= d2.sum(0); W1 -= X.T @ dH; b1 -= dH.sum(0)
    res["XOR_MLP"] = dict(acc=float(((P[:, 0] > .5) == y).mean()), pred=P[:, 0].round(3).tolist())
    gx, gy = np.meshgrid(np.linspace(-.5, 1.5, 200), np.linspace(-.5, 1.5, 200))
    grid = np.c_[gx.ravel(), gy.ravel()]
    pp = Perceptron(2)
    pp.fit(X, y, 200)
    Pm = sg(np.tanh(grid @ W1 + b1) @ W2 + b2).reshape(gx.shape)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a in ax:
        a.scatter(X[y == 0, 0], X[y == 0, 1], c=B, s=90, label="lớp 0")
        a.scatter(X[y == 1, 0], X[y == 1, 1], c=R, s=90, label="lớp 1")
        a.set_xlim(-.5, 1.5); a.set_ylim(-.5, 1.5); a.set_xlabel("x₁"); a.set_ylabel("x₂")
    ax[1].contourf(gx, gy, pp.predict(grid).reshape(gx.shape), levels=[-.5, .5, 1.5], colors=[B, R], alpha=.15)
    ax[2].contourf(gx, gy, Pm, levels=[0, .5, 1], colors=[B, R], alpha=.15)
    ax[0].legend(fontsize=8)
    ax[0].set_title("A. Dữ liệu XOR")
    ax[1].set_title("B. Perceptron (1958): đúng %.0f%%" % (100 * (pp.predict(X) == y).mean()))
    ax[2].set_title("C. MLP 2-4-1 (1986): đúng %.0f%%" % (100 * res["XOR_MLP"]["acc"]))
    plt.tight_layout(); plt.savefig(FIG + "c1_xor.png", bbox_inches="tight")
    out = res

elif which == "iris":
    from sklearn.datasets import load_iris
    d = load_iris(as_frame=True)
    df = d.frame
    df["species"] = df["target"].map(dict(enumerate(d.target_names)))
    out["shape"] = list(df.shape)
    out["counts"] = df["species"].value_counts().to_dict()
    out["describe"] = df.drop(columns=["target"]).groupby("species").mean(numeric_only=True).round(2).to_dict("index")
    out["head"] = df.sample(6, random_state=1).drop(columns=["target"]).round(1).to_dict("records")

    def perc(X, y, epochs=200):
        w = np.zeros(X.shape[1]); b = 0.; hist = []
        for _ in range(epochs):
            e = 0
            for xi, yi in zip(X, y):
                dlt = yi - (xi @ w + b > 0)
                w += .1 * dlt * xi; b += .1 * dlt; e += int(dlt != 0)
            hist.append(e)
            if e == 0:
                break
        return (X @ w + b > 0).astype(int), hist

    X = df.iloc[:, :4].to_numpy()
    yv = df["target"].to_numpy()
    p1, h1 = perc(X, (yv == 0).astype(int))
    m = yv > 0
    p2, h2 = perc(X[m], (yv[m] == 2).astype(int))
    out["perceptron"] = {"setosa_vs_rest": dict(acc=float((p1 == (yv == 0)).mean()), epochs=len(h1), converged=bool(h1[-1] == 0)),
                         "versicolor_vs_virginica": dict(acc=float((p2 == (yv[m] == 2)).mean()), epochs=len(h2), converged=bool(h2[-1] == 0))}
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    cols = [B, R, G]
    for k, nm in enumerate(d.target_names):
        ax[0].scatter(df.loc[yv == k, "petal length (cm)"], df.loc[yv == k, "petal width (cm)"], c=cols[k], s=18, label=nm)
        ax[1].scatter(df.loc[yv == k, "sepal length (cm)"], df.loc[yv == k, "sepal width (cm)"], c=cols[k], s=18)
    ax[0].set_xlabel("dài cánh hoa (cm)"); ax[0].set_ylabel("rộng cánh hoa (cm)"); ax[0].set_title("A. Đặc trưng cánh hoa"); ax[0].legend(fontsize=7.5)
    ax[1].set_xlabel("dài đài hoa (cm)"); ax[1].set_ylabel("rộng đài hoa (cm)"); ax[1].set_title("B. Đặc trưng đài hoa: chồng lấn")
    ax[2].bar(d.target_names, [50, 50, 50], color=cols); ax[2].set_ylim(0, 60); ax[2].set_title("C. Phân bố lớp: 50/50/50"); ax[2].set_ylabel("số mẫu")
    plt.tight_layout(); plt.savefig(FIG + "c1_iris.png", bbox_inches="tight")

elif which == "mnist":
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    z = np.load("mnist.npz")
    X = z["X"].astype("float32") / 255
    y = z["y"].astype(int)
    Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
    out["shape"] = [70000, 28, 28]
    out["train_counts"] = np.bincount(ytr).tolist()
    out["test_counts"] = np.bincount(yte).tolist()
    out["pixel_mean"] = float(X.mean())
    out["pixel_nonzero"] = float((X > 0).mean())
    fig, ax = plt.subplots(2, 10, figsize=(11, 2.7))
    for k in range(10):
        idx = np.where(ytr == k)[0][:2]
        for r in range(2):
            ax[r, k].imshow(Xtr[idx[r]].reshape(28, 28), cmap="gray_r")
            ax[r, k].axis("off")
        ax[0, k].set_title(str(k), fontsize=9)
    plt.tight_layout(); plt.savefig(FIG + "c1_mnist_samples.png", bbox_inches="tight"); plt.close()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar(range(10), out["train_counts"], color=B)
    ax.set_xticks(range(10)); ax.set_xlabel("chữ số"); ax.set_ylabel("số ảnh (tập train)")
    for i, v in enumerate(out["train_counts"]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_title("Phân bố lớp MNIST: gần cân bằng")
    plt.tight_layout(); plt.savefig(FIG + "c1_mnist_dist.png", bbox_inches="tight"); plt.close()
    rows = []

    def rec(name, era, pred, sec):
        rows.append(dict(model=name, era=era, err=float((pred != yte).mean() * 100), sec=round(sec, 1)))
        print(rows[-1], flush=True)

    t = time.perf_counter(); lr = LogisticRegression(max_iter=100).fit(Xtr, ytr)
    rec("Hồi quy logistic (tuyến tính)", "1950-60", lr.predict(Xte), time.perf_counter() - t)
    t = time.perf_counter(); kn = KNeighborsClassifier(3, n_jobs=4).fit(Xtr, ytr)
    rec("k láng giềng gần nhất (k=3)", "1960-90", kn.predict(Xte), time.perf_counter() - t)
    t = time.perf_counter(); sv = SVC(kernel="rbf", C=5, gamma="scale").fit(Xtr, ytr)
    rec("SVM nhân RBF", "1990-2000", sv.predict(Xte), time.perf_counter() - t)
    torch.manual_seed(0); torch.set_num_threads(4)

    def train(net, ep, lr_, flat):
        opt = torch.optim.Adam(net.parameters(), lr=lr_)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
        t0 = time.perf_counter()
        for e in range(ep):
            perm = torch.randperm(60000)
            for i in range(0, 60000, 128):
                b = perm[i:i + 128]
                xb = Xt[b] if flat else Xt[b].view(-1, 1, 28, 28)
                opt.zero_grad(); nn.functional.cross_entropy(net(xb), yt[b]).backward(); opt.step()
        with torch.no_grad():
            xe = torch.tensor(Xte) if flat else torch.tensor(Xte).view(-1, 1, 28, 28)
            p = net(xe).argmax(1).numpy()
        return p, time.perf_counter() - t0

    mlp = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    p, s = train(mlp, 10, 1e-3, True); rec("MLP 784-128-10 (lan truyền ngược)", "1986", p, s)
    cnn = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(), nn.AvgPool2d(2), nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
                        nn.Flatten(), nn.Linear(400, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))
    p, s = train(cnn, 10, 2e-3, False); rec("CNN kiểu LeNet-5", "1998", p, s)
    out["rows"] = rows
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    names = [r["model"].split(" (")[0] for r in rows]
    errs = [r["err"] for r in rows]
    ax.barh(range(len(rows)), errs, color=[B, B, B, O, R])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    for i, v in enumerate(errs):
        ax.text(v, i, " %.2f%%" % v, va="center", fontsize=8)
    ax.set_xlabel("Sai số trên 10.000 ảnh test (%)"); ax.set_title("Cùng một bài toán, các thế hệ phương pháp khác nhau"); ax.set_xlim(0, max(errs) * 1.2)
    plt.tight_layout(); plt.savefig(FIG + "c1_mnist_eras.png", bbox_inches="tight")

elif which == "diabetes":
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    D = np.load("ch2_diabetes.npz")
    Xtr, Xte, ytr, yte, Xr = D["X_tr"], D["X_te"], D["y_tr"].ravel(), D["y_te"].ravel(), D["Xraw_te"]
    feats = list(D["feats"])
    hb, gl = feats.index("HbA1c_level"), feats.index("blood_glucose_level")

    def met(y, p, s=None):
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        return dict(accuracy=float((p == y).mean()), precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9),
                    auc=(float(roc_auc_score(y, s)) if s is not None else None))
    rows = []
    rule = ((Xr[:, hb] >= 6.5) | (Xr[:, gl] >= 200)).astype(int)
    rows.append(dict(model="Luật chuyên gia: HbA1c ≥ 6,5 hoặc đường huyết ≥ 200", era="hệ chuyên gia 1970-80", **met(yte, rule)))
    rule2 = (Xr[:, hb] >= 6.5).astype(int)
    rows.append(dict(model="Luật chuyên gia chỉ dùng HbA1c ≥ 6,5", era="hệ chuyên gia 1970-80", **met(yte, rule2)))
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    rows.append(dict(model="Hồi quy logistic", era="thống kê, ML cổ điển", **met(yte, lr.predict(Xte), lr.predict_proba(Xte)[:, 1])))
    dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xtr, ytr)
    rows.append(dict(model="Cây quyết định (độ sâu 5)", era="ML cổ điển 1980-90", **met(yte, dt.predict(Xte), dt.predict_proba(Xte)[:, 1])))
    mlp = MLPClassifier((32, 16, 8), max_iter=200, random_state=42).fit(Xtr, ytr)
    rows.append(dict(model="MLP 32-16-8", era="học sâu 2010+", **met(yte, mlp.predict(Xte), mlp.predict_proba(Xte)[:, 1])))
    out["rows"] = rows
    df = pd.read_csv("../ch2/data/diabetes_prediction_dataset.csv")
    out["raw_shape"] = list(df.shape)
    out["raw_pos"] = int(df["diabetes"].sum())
    out["head"] = df.head(5).to_dict("records")
    out["describe"] = df.describe().round(2).to_dict()
    out["smoking_counts"] = df["smoking_history"].value_counts().to_dict()
    out["gender_counts"] = df["gender"].value_counts().to_dict()
    n1 = out["raw_pos"]; n0 = len(df) - n1
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].bar(["Không bệnh", "Tiểu đường"], [n0, n1], color=[B, R])
    ax[0].set_title("A. Nhãn: %.1f%% / %.1f%%" % (100 * n0 / len(df), 100 * n1 / len(df))); ax[0].set_ylabel("số hồ sơ")
    for v, x in zip([n0, n1], [0, 1]):
        ax[0].text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    for lab, c in [(0, B), (1, R)]:
        ax[1].hist(df.loc[df["diabetes"] == lab, "HbA1c_level"], bins=30, alpha=.6, color=c, label=["không bệnh", "tiểu đường"][lab], density=True)
        ax[2].hist(df.loc[df["diabetes"] == lab, "age"], bins=30, alpha=.6, color=c, density=True)
    ax[1].axvline(6.5, c="k", ls="--", lw=1)
    ax[1].set_title("B. HbA1c (ngưỡng chẩn đoán 6,5)"); ax[1].set_xlabel("HbA1c (%)"); ax[1].legend(fontsize=8)
    ax[2].set_title("C. Tuổi"); ax[2].set_xlabel("tuổi")
    plt.tight_layout(); plt.savefig(FIG + "c1_diabetes.png", bbox_inches="tight")

json.dump(out, open(path, "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print("saved", path)
