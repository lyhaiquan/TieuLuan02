# -*- coding: utf-8 -*-
"""Cung mot MLP, cung sieu tham so, cai dat ba lan: NumPy (dl_scratch) / PyTorch / Keras.
Chay: python ch2_run.py numpy|torch|keras   -> results/ch2_<fw>.json"""
import json, sys, time, os
import numpy as np

FW = sys.argv[1] if len(sys.argv) > 1 else None
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else ["diabetes", "housing", "comments"]
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "42,43,44").split(","))
CFG = {  # kien truc tot nhat cua tung he thong trong Assignment 03; lr/epochs/batch giu nguyen
    "diabetes": dict(sizes=[8, 32, 16, 8, 1], task="binary", lr=0.1, epochs=1500, batch=None),
    "housing": dict(sizes=[15, 128, 64, 1], task="regression", lr=0.02, epochs=120, batch=1024),
    "comments": dict(sizes=[1000, 32, 3], task="multiclass", lr=0.15, epochs=200, batch=256),
}

def sig(z): return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

def metrics(name, D, out):
    """out: xac suat (binary), gia tri chuan hoa (regression), xac suat (N,3) (multiclass)."""
    from sklearn.metrics import roc_auc_score
    if name == "diabetes":
        p = np.asarray(out).reshape(-1); y = D["y_te"].reshape(-1)
        yh = (p >= .5).astype(int); tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        return dict(accuracy=float((yh == y).mean()), precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12), roc_auc=float(roc_auc_score(y, p)), fn=fn)
    if name == "housing":
        z = np.asarray(out).reshape(-1); y = D["y_te"].reshape(-1)
        r2z = 1 - ((z - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        price = np.exp(z * float(D["y_sd"]) + float(D["y_mu"])); pt = D["price_te"]
        r2p = 1 - ((price - pt) ** 2).sum() / ((pt - pt.mean()) ** 2).sum()
        return dict(r2_log=float(r2z), r2_price=float(r2p), mae=float(np.abs(price - pt).mean()), rmse=float(np.sqrt(((price - pt) ** 2).mean())), mape=float((np.abs(price - pt) / pt).mean()))
    P = np.asarray(out); y = D["y_te"].astype(int); yh = P.argmax(1); f = []
    for k in range(3):
        tp = ((yh == k) & (y == k)).sum(); fp = ((yh == k) & (y != k)).sum(); fn = ((yh != k) & (y == k)).sum()
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1); f.append(2 * p_ * r_ / max(p_ + r_, 1e-12))
    return dict(accuracy=float((yh == y).mean()), macro_f1=float(np.mean(f)), f1_neg=float(f[0]), f1_neu=float(f[1]), f1_pos=float(f[2]))

def run_numpy(name, D, c, seed):
    import dl_scratch as dl
    Y = D["y_tr"]; Y = dl.one_hot(Y, 3) if c["task"] == "multiclass" else Y
    m = dl.MLPScratch(c["sizes"], c["task"], lr=c["lr"], epochs=c["epochs"], seed=seed)
    t0 = time.perf_counter(); m.fit(D["X_tr"].astype(float), Y, batch_size=c["batch"]); sec = time.perf_counter() - t0
    return metrics(name, D, m.predict_proba(D["X_te"].astype(float))), sec, m.n_params, m.loss_history

def run_torch(name, D, c, seed):
    import torch, torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(seed)
    layers = []
    for i, (a, b) in enumerate(zip(c["sizes"][:-1], c["sizes"][1:])):
        lin = nn.Linear(a, b); nn.init.kaiming_normal_(lin.weight, nonlinearity="relu"); nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(c["sizes"]) - 2: layers.append(nn.ReLU())
    net = nn.Sequential(*layers); opt = torch.optim.SGD(net.parameters(), lr=c["lr"])
    X = torch.tensor(D["X_tr"], dtype=torch.float32)
    y = torch.tensor(D["y_tr"], dtype=torch.long if c["task"] == "multiclass" else torch.float32)
    if c["task"] == "multiclass": y = y.view(-1)
    lossf = {"binary": nn.BCEWithLogitsLoss(), "regression": lambda o, t: 0.5 * nn.functional.mse_loss(o, t), "multiclass": nn.CrossEntropyLoss()}[c["task"]]
    g = torch.Generator().manual_seed(seed); N = len(X); bs = c["batch"] or N; hist = []
    t0 = time.perf_counter()
    for ep in range(c["epochs"]):
        perm = torch.randperm(N, generator=g) if bs < N else torch.arange(N); tot = 0.0; nb = 0
        for s in range(0, N, bs):
            idx = perm[s:s + bs]; opt.zero_grad(); loss = lossf(net(X[idx]), y[idx]); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        hist.append(tot / nb)
    sec = time.perf_counter() - t0
    with torch.no_grad():
        o = net(torch.tensor(D["X_te"], dtype=torch.float32))
        out = torch.sigmoid(o).numpy() if c["task"] == "binary" else (torch.softmax(o, 1).numpy() if c["task"] == "multiclass" else o.numpy())
    return metrics(name, D, out), sec, sum(p.numel() for p in net.parameters()), hist

def run_keras(name, D, c, seed):
    os.environ["KERAS_BACKEND"] = "tensorflow"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import keras
    from keras import layers
    keras.utils.set_random_seed(seed)
    acts = ["relu"] * (len(c["sizes"]) - 2) + [{"binary": "sigmoid", "regression": None, "multiclass": "softmax"}[c["task"]]]
    net = keras.Sequential([keras.Input((c["sizes"][0],))] + [layers.Dense(b, activation=a, kernel_initializer="he_normal", bias_initializer="zeros") for b, a in zip(c["sizes"][1:], acts)])
    loss = {"binary": "binary_crossentropy", "regression": lambda t, p: 0.5 * keras.ops.mean(keras.ops.square(p - t)), "multiclass": "sparse_categorical_crossentropy"}[c["task"]]
    net.compile(optimizer=keras.optimizers.SGD(c["lr"]), loss=loss)
    bs = c["batch"] or len(D["X_tr"])
    t0 = time.perf_counter(); h = net.fit(D["X_tr"].astype("float32"), D["y_tr"].astype("float32" if c["task"] != "multiclass" else "int32"), batch_size=bs, epochs=c["epochs"], shuffle=True, verbose=0); sec = time.perf_counter() - t0
    out = net.predict(D["X_te"].astype("float32"), batch_size=4096, verbose=0)
    return metrics(name, D, out), sec, net.count_params(), h.history["loss"]

def main():
    fn = dict(numpy=run_numpy, torch=run_torch, keras=run_keras)[FW]
    path = "results/ch2_%s%s.json" % (FW, os.environ.get("SUFFIX", ""))
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in ONLY:
        D = dict(np.load("data/ch2_%s.npz" % name)); c = CFG[name]; runs = []
        for seed in SEEDS:
            m, sec, npar, hist = fn(name, D, c, seed)
            runs.append(dict(seed=seed, metrics=m, seconds=round(sec, 2), params=int(npar), loss=[round(float(v), 6) for v in hist]))
            print(FW, name, seed, {k: round(v, 4) for k, v in m.items()}, "%.1fs" % sec, flush=True)
        keys = runs[0]["metrics"].keys()
        res[name] = dict(config=c, runs=runs, mean={k: float(np.mean([r["metrics"][k] for r in runs])) for k in keys}, std={k: float(np.std([r["metrics"][k] for r in runs])) for k in keys}, seconds=float(np.mean([r["seconds"] for r in runs])))
        json.dump(res, open(path, "w"), indent=1)
    print("DONE", FW)


if __name__ == "__main__":
    main()
