# -*- coding: utf-8 -*-
"""RNN / GRU / LSTM viet bang NumPy thuan: forward, BPTT, Adam, cat gradient, dung som.

Quy uoc trong so giong het torch.nn de co the doi chieu tung so:
    W_ih (G*H, F), W_hh (G*H, H), b_ih (G*H), b_hh (G*H)
    G = 1 (RNN), 3 (GRU: r, z, n), 4 (LSTM: i, f, g, o)
Mang nhieu-mot: lay trang thai an o buoc cuoi, (dropout), qua mot lop Linear.
"""
import copy
import time

import numpy as np

GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class Recurrent:
    def __init__(self, cell, n_in, H, rng):
        self.cell, self.H, self.F, self.G = cell, H, n_in, GATES[cell]
        k = 1.0 / np.sqrt(H)
        GH = self.G * H
        self.p = {"W_ih": rng.uniform(-k, k, (GH, n_in)), "W_hh": rng.uniform(-k, k, (GH, H)),
                  "b_ih": rng.uniform(-k, k, GH), "b_hh": rng.uniform(-k, k, GH)}

    # ------------------------------------------------------------ forward
    def forward(self, X):
        """X: (B, T, F) -> h_T: (B, H) va cache cho BPTT."""
        B, T, _ = X.shape
        H, p = self.H, self.p
        XW = X @ p["W_ih"].T + p["b_ih"]                  # (B, T, GH): mot phep nhan cho ca chuoi
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = []
        for t in range(T):
            gh = h @ p["W_hh"].T + p["b_hh"]
            if self.cell == "RNN":
                h_new = np.tanh(XW[:, t] + gh)
                cache.append((h, h_new))
            elif self.cell == "GRU":
                gi = XW[:, t]
                r = sigm(gi[:, :H] + gh[:, :H])
                z = sigm(gi[:, H:2 * H] + gh[:, H:2 * H])
                n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
                h_new = (1 - z) * n + z * h
                cache.append((h, r, z, n, gh[:, 2 * H:]))
            else:
                g_all = XW[:, t] + gh
                i = sigm(g_all[:, :H])
                f = sigm(g_all[:, H:2 * H])
                g = np.tanh(g_all[:, 2 * H:3 * H])
                o = sigm(g_all[:, 3 * H:])
                c_new = f * c + i * g
                tc = np.tanh(c_new)
                h_new = o * tc
                cache.append((h, c, i, f, g, o, tc))
                c = c_new
            h = h_new
        self._X = X
        self._cache = cache
        return h

    # ----------------------------------------------------------- backward
    def backward(self, dh):
        """dh: dL/dh_T (B, H) -> gradient cac tham so (BPTT), va dL/dX."""
        X, cache, p, H = self._X, self._cache, self.p, self.H
        B, T, _ = X.shape
        GH = self.G * H
        dXW = np.zeros((B, T, GH))
        dW_hh = np.zeros_like(p["W_hh"])
        db_hh = np.zeros(GH)
        dc = np.zeros((B, H))
        for t in range(T - 1, -1, -1):
            if self.cell == "RNN":
                h_prev, h_new = cache[t]
                da = dh * (1 - h_new ** 2)
                dgi, dgh = da, da
                dh = da @ p["W_hh"]
            elif self.cell == "GRU":
                h_prev, r, z, n, hn = cache[t]
                dn = dh * (1 - z)
                dz = dh * (h_prev - n)
                dh_prev = dh * z
                dn_pre = dn * (1 - n ** 2)
                dr_pre = dn_pre * hn * r * (1 - r)
                dz_pre = dz * z * (1 - z)
                dgi = np.concatenate([dr_pre, dz_pre, dn_pre], axis=1)
                dgh = np.concatenate([dr_pre, dz_pre, dn_pre * r], axis=1)
                dh = dh_prev + dgh @ p["W_hh"]
            else:
                h_prev, c_prev, i, f, g, o, tc = cache[t]
                do = dh * tc
                dc = dc + dh * o * (1 - tc ** 2)
                di, dg, df = dc * g, dc * i, dc * c_prev
                dc = dc * f
                dgi = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                      dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
                dgh = dgi
                dh = dgh @ p["W_hh"]
            dXW[:, t] = dgi
            dW_hh += dgh.T @ h_prev
            db_hh += dgh.sum(0)
        dXW2 = dXW.reshape(-1, GH)
        grads = {"W_ih": dXW2.T @ X.reshape(-1, X.shape[2]), "W_hh": dW_hh,
                 "b_ih": dXW2.sum(0), "b_hh": db_hh}
        return grads, dXW @ p["W_ih"]


class SeqNetNP:
    """Recurrent -> (Dropout) -> Linear(H, 1), cung hinh dang voi SeqNet cua PyTorch."""

    def __init__(self, cell, n_in, H, dropout=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.rnn = Recurrent(cell, n_in, H, rng)
        k = 1.0 / np.sqrt(H)
        self.p = {"W_o": rng.uniform(-k, k, (1, H)), "b_o": rng.uniform(-k, k, 1)}
        self.dropout, self.rng, self.H = dropout, rng, H

    def params(self):
        return {**{"rnn." + k: v for k, v in self.rnn.p.items()}, **self.p}

    def n_params(self):
        return int(sum(v.size for v in self.params().values()))

    def forward(self, X, train=False):
        h = self.rnn.forward(X)
        if train and self.dropout > 0:
            self._mask = (self.rng.random(h.shape) >= self.dropout) / (1 - self.dropout)
            h = h * self._mask
        else:
            self._mask = None
        self._h = h
        return (h @ self.p["W_o"].T + self.p["b_o"]).ravel()

    def backward(self, dout):
        """dout: dL/d(logit) (B,)."""
        d = dout[:, None]
        g = {"W_o": d.T @ self._h, "b_o": d.sum(0)}
        dh = d @ self.p["W_o"]
        if self._mask is not None:
            dh = dh * self._mask
        gr, _ = self.rnn.backward(dh)
        return {**{"rnn." + k: v for k, v in gr.items()}, **g}

    def set_params(self, d):
        for k, v in d.items():
            (self.rnn.p if k.startswith("rnn.") else self.p)[k.split(".")[-1]][...] = v

    def get_params(self):
        return {k: v.copy() for k, v in self.params().items()}


def bce_grad(logit, y):
    p = sigm(logit)
    loss = np.mean(np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit))))
    return loss, (p - y) / len(y)


def mse_grad(pred, y):
    e = pred - y
    return float(np.mean(e ** 2)), 2 * e / len(y)


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global(grads, max_norm=1.0):
    n = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if n > max_norm:
        s = max_norm / (n + 1e-6)
        for k in grads:
            grads[k] = grads[k] * s
    return n


def fit(model, data, loss_grad, hp, seed):
    """Adam + cat chuan gradient 1.0 + dung som theo loss validation (giong ban PyTorch)."""
    rng = np.random.default_rng(seed)
    Xtr, ytr = data["X_train"].astype(np.float64), data["y_train"].astype(np.float64)
    Xva, yva = data["X_val"].astype(np.float64), data["y_val"].astype(np.float64)
    params = model.params()
    opt = Adam(params, hp["lr"])
    hist = {"train_loss": [], "val_loss": []}
    best, best_state, wait = np.inf, None, 0
    t0 = time.perf_counter()
    for epoch in range(hp["epochs"]):
        perm = rng.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), hp["batch"]):
            b = perm[i:i + hp["batch"]]
            out = model.forward(Xtr[b], train=True)
            loss, dout = loss_grad(out, ytr[b])
            grads = model.backward(dout)
            clip_global(grads, 1.0)
            opt.step(params, grads)
            total += loss * len(b)
        vl = loss_grad(model.forward(Xva), yva)[0]
        hist["train_loss"].append(total / len(Xtr))
        hist["val_loss"].append(float(vl))
        if vl < best - 1e-6:
            best, best_state, wait = vl, model.get_params(), 0
        else:
            wait += 1
            if wait >= hp["patience"]:
                break
    model.set_params(best_state)
    return dict(history=hist, epochs=epoch + 1, best_epoch=int(np.argmin(hist["val_loss"])) + 1,
                seconds=round(time.perf_counter() - t0, 2))


class MLPNP:
    """Doi chung khong hoi quy: Flatten -> Linear(T, H) -> ReLU -> Linear(H, 1), giong MLP cua PyTorch."""

    def __init__(self, T, H, seed=0):
        rng = np.random.default_rng(seed)
        k1, k2 = 1.0 / np.sqrt(T), 1.0 / np.sqrt(H)
        self.p = {"W1": rng.uniform(-k1, k1, (T, H)), "b1": rng.uniform(-k1, k1, H),
                  "W2": rng.uniform(-k2, k2, (H, 1)), "b2": rng.uniform(-k2, k2, 1)}

    def params(self):
        return self.p

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def forward(self, X, train=False):
        self._x = X.reshape(len(X), -1)
        self._z = self._x @ self.p["W1"] + self.p["b1"]
        self._h = np.maximum(self._z, 0)
        return (self._h @ self.p["W2"] + self.p["b2"]).ravel()

    def backward(self, dout):
        d = dout[:, None]
        dh = (d @ self.p["W2"].T) * (self._z > 0)
        return {"W1": self._x.T @ dh, "b1": dh.sum(0), "W2": self._h.T @ d, "b2": d.sum(0)}

    def get_params(self):
        return {k: v.copy() for k, v in self.p.items()}

    def set_params(self, d):
        for k, v in d.items():
            self.p[k][...] = v
