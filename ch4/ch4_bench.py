# -*- coding: utf-8 -*-
"""Do thoi gian mot epoch (giay) cua RNN/LSTM/GRU o ba cai dat, chay tuan tu tren may roi.
python ch4_bench.py numpy|torch|keras  ->  results/bench_<fw>.json"""
import json, os, sys, time
import numpy as np
import lab06_data as D

FW = sys.argv[1]
EPOCHS = 3
out = {}
C = D.churn_splits()
SV = D.stock_splits("y_vol")
tasks = {"churn": (C["X_train"][:, :, None].astype("float32"), C["y_train"], 3e-3, 0.0, "bce"),
         "vol": (SV["X_train"], SV["y_train"], 1e-3, 0.2, "mse")}
H, BATCH = 32, 64


def run_numpy(cell, X, y, lr, drop, loss):
    import rnn_numpy as R
    m = R.SeqNetNP(cell, X.shape[2], H, drop, 0)
    params = m.params(); opt = R.Adam(params, lr); X = X.astype(float); y = y.astype(float)
    lg = R.bce_grad if loss == "bce" else R.mse_grad
    rng = np.random.default_rng(0); times = []
    for ep in range(EPOCHS + 1):
        t0 = time.perf_counter(); perm = rng.permutation(len(X))
        for i in range(0, len(X), BATCH):
            b = perm[i:i + BATCH]; o = m.forward(X[b], True); _, d = lg(o, y[b]); g = m.backward(d); R.clip_global(g, 1.0); opt.step(params, g)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times[1:]))


def run_torch(cell, X, y, lr, drop, loss):
    import torch, torch.nn as nn
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell](X.shape[2], H, batch_first=True); s.drop = nn.Dropout(drop); s.head = nn.Linear(H, 1)

        def forward(s, x):
            return s.head(s.drop(s.rnn(x)[0][:, -1])).squeeze(-1)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss() if loss == "bce" else nn.MSELoss()
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y); times = []
    for ep in range(EPOCHS + 1):
        t0 = time.perf_counter(); perm = torch.randperm(len(Xt)); net.train()
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]; opt.zero_grad(); lf(net(Xt[b]), yt[b]).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times[1:]))


def run_keras(cell, X, y, lr, drop, loss):
    os.environ["KERAS_BACKEND"] = "tensorflow"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import keras
    from keras import layers
    keras.utils.set_random_seed(0)
    L = {"RNN": layers.SimpleRNN, "LSTM": layers.LSTM, "GRU": layers.GRU}[cell]
    net = keras.Sequential([keras.Input((X.shape[1], X.shape[2])), L(H)] + ([layers.Dropout(drop)] if drop else []) + [layers.Dense(1)])
    net.compile(optimizer=keras.optimizers.Adam(lr, global_clipnorm=1.0), loss=keras.losses.BinaryCrossentropy(from_logits=True) if loss == "bce" else "mse")
    net.fit(X, y, batch_size=BATCH, epochs=1, verbose=0)
    t0 = time.perf_counter(); net.fit(X, y, batch_size=BATCH, epochs=EPOCHS, verbose=0)
    return (time.perf_counter() - t0) / EPOCHS


fn = dict(numpy=run_numpy, torch=run_torch, keras=run_keras)[FW]
for task, (X, y, lr, drop, loss) in tasks.items():
    for cell in ("RNN", "LSTM", "GRU"):
        out["%s__%s" % (task, cell)] = fn(cell, X, y, lr, drop, loss)
        print(FW, task, cell, round(out["%s__%s" % (task, cell)], 3), flush=True)
json.dump(out, open("results/bench_%s.json" % FW, "w"), indent=1)
