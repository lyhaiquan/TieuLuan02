# -*- coding: utf-8 -*-
"""
dl_scratch.py -- Mang no-ron sau viet tu dau, 100% NumPy thuan.

KHONG dung TensorFlow / PyTorch / Keras, va KHONG dung scikit-learn cho
phan mang no-ron. Toan bo forward, loss, backward va gradient descent
deu tu viet.

Module ho tro 3 loai bai toan (3 he thong trong Assignment 03):
    - 'binary'     : phan loai nhi phan  -> Sigmoid + Binary Cross-Entropy
    - 'regression' : hoi quy lien tuc    -> Linear  + Mean Squared Error
    - 'multiclass' : phan loai da lop    -> Softmax + Categorical Cross-Entropy

Kien truc goc trong bai giang la 8 -> 16 -> 8 -> 1. Module nay tong quat hoa
thanh mot danh sach kich thuoc tuy y, vi du [14, 64, 32, 16, 1].
"""

import numpy as np

EPS = 1e-9  # chan duoi chong log(0)


# =====================================================================
# 1. HAM KICH HOAT
# =====================================================================

def relu(z):
    """ReLU(z) = max(0, z), ap dung tren tung phan tu."""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """ReLU'(z) = 1 neu z > 0, nguoc lai 0.

    Luu y: tham so truyen vao la Z (truoc kich hoat), KHONG phai H (sau
    kich hoat). Voi ReLU thi (H > 0) va (Z > 0) trung nhau nen de nham,
    nhung voi cac ham kich hoat khac thi sai hoan toan.
    """
    return (z > 0.0).astype(float)


def sigmoid(z):
    """Sigmoid on dinh so hoc.

    Cong thuc trong slide 1/(1+exp(-z)) bi tran so khi z rat am
    (exp(-z) -> vo cung). Tach lam hai nhanh de exp() luon nhan doi so <= 0:
        z >= 0 : 1 / (1 + e^-z)
        z <  0 : e^z / (1 + e^z)
    Hai nhanh bang nhau ve mat toan hoc nhung nhanh duoi khong bao gio tran.
    """
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


def softmax(z):
    """Softmax theo hang, on dinh so hoc.

    Tru di max cua moi hang truoc khi lay exp: softmax(z) = softmax(z - c)
    voi moi hang so c, nen phep tru nay khong doi ket qua nhung dua so mu
    lon nhat ve e^0 = 1, triet tieu hoan toan kha nang tran so.
    """
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# =====================================================================
# 2. HAM MAT MAT
# =====================================================================

def binary_cross_entropy(y_true, y_pred):
    """L = -(1/N) * sum[ y*log(p) + (1-y)*log(1-p) ]"""
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def mse_loss(y_true, y_pred):
    """L = (1/2N) * sum (y_hat - y)^2.

    He so 1/2 lam dao ham triet tieu thanh (y_hat - y)/N, dong dang voi
    dang rut gon cua Sigmoid+BCE va Softmax+CCE.
    """
    return float(np.mean((y_pred - y_true) ** 2) / 2.0)


def categorical_cross_entropy(y_onehot, y_pred):
    """L = -(1/N) * sum_i sum_k y_ik * log(p_ik)"""
    p = np.clip(y_pred, EPS, 1.0)
    return float(-np.mean(np.sum(y_onehot * np.log(p), axis=1)))


LOSS_FN = {
    "binary": binary_cross_entropy,
    "regression": mse_loss,
    "multiclass": categorical_cross_entropy,
}


# =====================================================================
# 3. MANG NO-RON
# =====================================================================

def dense_layer(A, W, b):
    """Mot tang tuyen tinh (affine): Z = A @ W + b."""
    return A @ W + b


class MLPScratch:
    """Mang no-ron nhieu tang viet tu dau bang NumPy.

    Tham so
    -------
    sizes : list[int]
        Kich thuoc cac tang, vi du [8, 16, 8, 1]. Phan tu dau la so dac
        trung dau vao, phan tu cuoi la so no-ron dau ra.
    task : {'binary', 'regression', 'multiclass'}
        Quyet dinh ham kich hoat tang cuoi va ham mat mat.
    lr : float
        Toc do hoc eta.
    epochs : int
        So chu ky duyet toan bo tap huan luyen.
    use_relu : bool
        Neu False thi bo toan bo ReLU o cac tang an (phuc vu Exercise 5 --
        chung minh xep chong tang tuyen tinh khong tang nang luc bieu dien).
    seed : int
        Hat giong ngau nhien, co dinh de tai lap 100%.
    """

    def __init__(self, sizes, task="binary", lr=0.1, epochs=1000,
                 use_relu=True, seed=42):
        assert task in LOSS_FN, f"task khong hop le: {task}"
        self.sizes = list(sizes)
        self.task = task
        self.lr = lr
        self.epochs = epochs
        self.use_relu = use_relu
        self.seed = seed
        self.params = self._init_params()
        self.loss_history = []
        self._rng = np.random.RandomState(seed + 1)   # dung rieng cho xao tron mini-batch

    # ---------------- khoi tao ----------------

    def _init_params(self):
        """Khoi tao He Normal: W ~ N(0, sqrt(2/n_in)), b = 0.

        W phai ngau nhien: neu moi W deu bang 0 thi moi no-ron trong cung
        mot tang nhan cung gradient, hoc y het nhau va tang 16 no-ron chi
        con gia tri cua 1 no-ron (symmetry breaking).
        b duoc phep bang 0 vi tinh doi xung da bi pha vo boi W.
        He so sqrt(2/n_in) giu phuong sai tin hieu on dinh khi di qua ReLU
        (ReLU cat bo mot nua phan phoi nen can nhan buc lai he so 2).
        """
        rng = np.random.RandomState(self.seed)
        params = []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            W = rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            params.append([W, b])
        return params

    @property
    def n_params(self):
        """Tong so tham so hoc duoc (dung cho bang so sanh kien truc)."""
        return int(sum(W.size + b.size for W, b in self.params))

    # ---------------- lan truyen tien ----------------

    def forward(self, X):
        """Tinh du doan va luu lai toan bo gia tri trung gian vao cache.

        Cache chinh la 'do thi tinh toan' (computational graph) thu cong:
        lan truyen nguoc can Z de tinh ReLU'(Z) va can A cua tang truoc
        de tinh dW. Neu khong luu, phai tinh lai tu dau.
        """
        A = X
        cache = {"A": [X], "Z": []}
        L = len(self.params)
        for i, (W, b) in enumerate(self.params):
            Z = dense_layer(A, W, b)
            if i == L - 1:                       # tang ra
                if self.task == "binary":
                    A = sigmoid(Z)
                elif self.task == "multiclass":
                    A = softmax(Z)
                else:                            # regression: kich hoat tuyen tinh
                    A = Z
            else:                                # tang an
                A = relu(Z) if self.use_relu else Z
            cache["Z"].append(Z)
            cache["A"].append(A)
        return A, cache

    # ---------------- lan truyen nguoc ----------------

    def backward(self, y_true, cache):
        """Tinh gradient bang quy tac day chuyen, duyet nguoc tu tang cuoi.

        Diem mau chot: voi ca ba cap (Sigmoid+BCE), (Softmax+CCE) va
        (Linear+MSE), dao ham cua loss theo Z cua TANG CUOI deu rut gon
        thanh cung mot bieu thuc:
                dZ_L = (y_hat - y) / N
        Day khong phai trung hop -- ca ba deu la truong hop rieng cua ho
        phan phoi mu voi ham lien ket chinh tac (canonical link).
        """
        N = y_true.shape[0]
        L = len(self.params)
        grads = [None] * L

        dZ = (cache["A"][-1] - y_true) / N

        for i in range(L - 1, -1, -1):
            A_prev = cache["A"][i]
            dW = A_prev.T @ dZ                      # (n_in, N) @ (N, n_out)
            db = np.sum(dZ, axis=0, keepdims=True)  # cong don sai so tren N mau
            grads[i] = (dW, db)
            if i > 0:
                dA = dZ @ self.params[i][0].T       # day gradient qua ma tran W
                if self.use_relu:
                    dZ = dA * relu_derivative(cache["Z"][i - 1])
                else:
                    dZ = dA
        return grads

    # ---------------- huan luyen ----------------

    def _step(self, Xb, yb, loss_fn):
        """Mot buoc cap nhat: Forward -> Loss -> Backward -> Update."""
        y_hat, cache = self.forward(Xb)
        loss = loss_fn(yb, y_hat)
        for i, (dW, db) in enumerate(self.backward(yb, cache)):
            # dau tru: di NGUOC huong gradient, vi gradient chi ve huong
            # ham mat mat TANG nhanh nhat
            self.params[i][0] -= self.lr * dW
            self.params[i][1] -= self.lr * db
        return loss

    def fit(self, X, y, batch_size=None, verbose_every=0):
        """Huan luyen bang Gradient Descent.

        batch_size = None  -> Full-batch GD, dung y het bai giang: moi epoch
                              duyet toan bo tap Train mot lan roi cap nhat
                              trong so DUNG MOT LAN.
        batch_size = k     -> Mini-batch GD: moi epoch chia ngau nhien tap
                              Train thanh cac lo k mau va cap nhat sau moi lo.
                              Gradient tren mot lo la uoc luong nhieu cua
                              gradient toan cuc, nhung mot epoch cho N/k lan
                              cap nhat thay vi 1, nen hoi tu nhanh hon HANG
                              CHUC LAN tren du lieu lon. Day chinh la mot
                              trong bay viec ma PyTorch/TensorFlow tu dong hoa.
        """
        loss_fn = LOSS_FN[self.task]
        self.loss_history = []
        N = X.shape[0]
        for ep in range(self.epochs):
            if not batch_size or batch_size >= N:
                loss = self._step(X, y, loss_fn)
            else:
                perm = self._rng.permutation(N)
                tot, nb = 0.0, 0
                for s in range(0, N, batch_size):
                    idx = perm[s:s + batch_size]
                    tot += self._step(X[idx], y[idx], loss_fn)
                    nb += 1
                loss = tot / nb
            self.loss_history.append(loss)
            if verbose_every and ep % verbose_every == 0:
                print(f"  epoch {ep:5d} | loss = {loss:.6f}", flush=True)
        return self

    # ---------------- suy luan ----------------

    def predict_proba(self, X):
        """Xac suat (binary/multiclass) hoac gia tri du doan (regression)."""
        return self.forward(X)[0]

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        if self.task == "binary":
            return (p >= threshold).astype(int)
        if self.task == "multiclass":
            return np.argmax(p, axis=1)
        return p

    def hidden_representations(self, X):
        """Tra ve [X, H1, H2, ...] de chieu PCA, quan sat hoc bieu dien."""
        _, cache = self.forward(X)
        return cache["A"][:-1]


# =====================================================================
# 4. KIEM CHUNG GRADIENT (gradient checking)
# =====================================================================

def gradient_check(model, X, y, n_samples=12, eps=1e-5, seed=0):
    """So sanh gradient giai tich voi sai phan huu han trung tam.

        dL/dw  ~=  ( L(w + eps) - L(w - eps) ) / (2*eps)

    Neu backward viet dung, sai so tuong doi phai o muc 1e-7 hoac nho hon.
    Cach nay chinh xac nhung cham gap hang nghin lan (moi tham so can 2
    luot forward), nen chi dung de kiem chung mot lan roi tat di.
    """
    rng = np.random.RandomState(seed)
    loss_fn = LOSS_FN[model.task]
    _, cache = model.forward(X)
    grads = model.backward(y, cache)

    rel_errors = []
    for _ in range(n_samples):
        li = rng.randint(len(model.params))
        W = model.params[li][0]
        r = rng.randint(W.shape[0])
        c = rng.randint(W.shape[1])

        old = W[r, c]
        W[r, c] = old + eps
        lp = loss_fn(y, model.forward(X)[0])
        W[r, c] = old - eps
        lm = loss_fn(y, model.forward(X)[0])
        W[r, c] = old

        numeric = (lp - lm) / (2 * eps)
        analytic = grads[li][0][r, c]
        denom = max(abs(numeric) + abs(analytic), 1e-12)
        rel_errors.append(abs(numeric - analytic) / denom)
    return float(np.max(rel_errors)), rel_errors


# =====================================================================
# 5. DO DO DANH GIA (tu viet, khong dung sklearn.metrics)
# =====================================================================

def binary_metrics(y_true, y_prob, threshold=0.5):
    """Accuracy / Precision / Recall / F1 + ma tran nham lan."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = (np.asarray(y_prob).reshape(-1) >= threshold).astype(int)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    # 'if ... else 0.0' chu khong cong 1e-8: khi mo hinh khong du doan
    # duong tinh nao thi Precision khong xac dinh, tra ve 0.0 la trung thuc
    # hon la mot so rat lon do chia cho 1e-8.
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": (TP + TN) / len(y_true), "precision": prec,
            "recall": rec, "f1": f1, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def roc_auc(y_true, y_score):
    """ROC-AUC tinh bang thong ke hang (Mann-Whitney U), khong can sklearn."""
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    s = y_score[order]
    i = 0
    while i < len(s):                      # xu ly diem so trung nhau: gan hang trung binh
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R^2 tren cung mot don vi voi y truyen vao."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan")}


def multiclass_metrics(y_true, y_prob, n_classes):
    """Accuracy + Precision/Recall/F1 tung lop + Macro F1."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.argmax(y_prob, axis=1)
    per = []
    for k in range(n_classes):
        TP = int(np.sum((y_pred == k) & (y_true == k)))
        FP = int(np.sum((y_pred == k) & (y_true != k)))
        FN = int(np.sum((y_pred != k) & (y_true == k)))
        p = TP / (TP + FP) if (TP + FP) else 0.0
        r = TP / (TP + FN) if (TP + FN) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per.append({"precision": p, "recall": r, "f1": f,
                    "support": int(np.sum(y_true == k))})
    return {"accuracy": float(np.mean(y_pred == y_true)),
            "macro_f1": float(np.mean([c["f1"] for c in per])),
            "per_class": per}


def confusion_matrix(y_true, y_pred, n_classes):
    """Ma tran nham lan n x n, hang = nhan that, cot = nhan du doan."""
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true).reshape(-1),
                    np.asarray(y_pred).reshape(-1)):
        M[int(t), int(p)] += 1
    return M


def one_hot(y, n_classes):
    """Chuyen nhan nguyen thanh ma tran one-hot (N, n_classes)."""
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), np.asarray(y).astype(int)] = 1.0
    return Y


# =====================================================================
# 6. TIEN XU LY TU VIET (chuan hoa, chia tap)
# =====================================================================

def stratified_split(y, test_size=0.2, seed=42):
    """Chia Train/Test giu nguyen ty le nhan o ca hai tap.

    Voi du lieu mat can bang (vi du 8.8% duong tinh), chia ngau nhien
    thuan tuy co the lam ty le nhan lech han giua hai tap, khien ket qua
    danh gia mat y nghia.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y).reshape(-1)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * (1 - test_size)))
        tr.append(idx[:k])
        te.append(idx[k:])
    tr = np.concatenate(tr)
    te = np.concatenate(te)
    rng.shuffle(tr)
    rng.shuffle(te)
    return tr, te


class StandardScalerScratch:
    """Chuan hoa Z-Score, fit CHI tren tap Train de chong ro ri du lieu."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0   # cot hang so: giu nguyen, khong chia 0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
