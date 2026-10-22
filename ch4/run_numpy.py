"""Huan luyen RNN/LSTM/GRU NumPy thuan tren Churn, AMZN loi suat, AMZN bien dong.
Dung dung lab06_data (chia tap, sieu tham so, do do) nhu ban PyTorch/Keras."""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import lab06_data as D
import rnn_numpy as R

SEEDS = (42, 43, 44)
task = sys.argv[1]                                   # churn | ret | vol
cells = sys.argv[2].split(",") if len(sys.argv) > 2 else ["RNN", "LSTM", "GRU"]

def summarize(runs, keys):
    return {k: [float(np.mean([r["metrics"][k] for r in runs])), float(np.std([r["metrics"][k] for r in runs]))] for k in keys}

def pack(runs, keys, extra):
    return dict(runs=[{k: v for k, v in r.items() if k not in ("p_test", "pred", "history")} for r in runs],
                summary=summarize(runs, keys), history=runs[0]["history"], **extra)

if task == "churn":
    C = D.churn_splits(); hp = D.CHURN_HP; H = hp["hidden"]
    seq = lambda X: X[:, :, None]
    data = dict(X_train=seq(C["X_train"]), y_train=C["y_train"], X_val=seq(C["X_val"]), y_val=C["y_val"])
    keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "accuracy_at_05"]
    for cell in cells:
        runs = []
        for seed in SEEDS:
            m = R.MLPNP(11, H, seed) if cell == "MLP" else R.SeqNetNP(cell, 1, H, 0.0, seed)
            info = R.fit(m, data, R.bce_grad, hp, seed)
            pv = R.sigm(m.forward(seq(C["X_val"]).astype(np.float64))); pt = R.sigm(m.forward(seq(C["X_test"]).astype(np.float64)))
            thr = D.best_threshold(C["y_val"], pv)
            r = dict(seed=seed, metrics=D.cls_metrics(C["y_test"], pt, thr), p_test=pt, **info)
            runs.append(r); print(cell, seed, r["metrics"]["roc_auc"], r["metrics"]["f1"], r["epochs"], r["seconds"], flush=True)
        D.save_result("numpy__churn__" + cell, pack(runs, keys, dict(framework="NumPy", task="churn", model=cell, params=m.n_params(), p_test=runs[0]["p_test"])))
else:
    S = D.stock_splits("y_ret" if task == "ret" else "y_vol"); hp = D.STOCK_HP
    keys = (["rmse", "mae", "r2_vs_zero", "dir_acc", "price_rmse", "price_mape"] if task == "ret" else ["rmse", "mae", "r2", "corr"])
    for cell in cells:
        runs = []
        for seed in SEEDS:
            m = R.SeqNetNP(cell, len(S["features"]), hp["hidden"], hp["dropout"], seed)
            info = R.fit(m, S, R.mse_grad, hp, seed)
            pred = D.unscale(S, m.forward(S["X_test"].astype(np.float64)))
            r = dict(seed=seed, metrics=D.reg_metrics(S, pred), pred=pred, **info)
            runs.append(r); print(task, cell, seed, r["metrics"], r["epochs"], r["seconds"], flush=True)
        D.save_result("numpy__%s__%s" % (task, cell), pack(runs, keys, dict(framework="NumPy", task=task, model=cell, params=m.n_params(), pred=runs[0]["pred"])))
print("DONE", task, cells)
