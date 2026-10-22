"""Du lieu, do do va baseline dung chung cho bai 06 (mang hoi quy RNN).

File nay chi dung numpy, pandas, scikit-learn de import duoc tu ca hai kernel:
"Python 3 (ipykernel)" chay PyTorch va "Python (TF 2.21)" chay Keras. Moi thu anh
huong toi ket qua so sanh ma khong phai mo hinh deu nam o day: cach chia tap, cach
chuan hoa, cua so thoi gian, sieu tham so, do do va cac baseline. Nho vay khi hai
framework cho so khac nhau, khac biet do den tu framework chu khong den tu du lieu.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42

# Sieu tham so dung chung: hai framework doc cung mot bang nay.
CHURN_HP = dict(hidden=32, lr=3e-3, batch=64, epochs=60, patience=10)
STOCK_HP = dict(hidden=32, lr=1e-3, batch=64, epochs=40, patience=8, dropout=0.2)


# ====================================================================== churn

CHURN_CSV = DATA / "Churn_Modelling.csv"
ID_COLS = ["RowNumber", "CustomerId", "Surname"]

# Thu tu RNN doc cac dac trung cua mot khach hang, moi buoc thoi gian mot dac trung:
# nhan khau hoc -> tai chinh -> quan he voi ngan hang. Muc 3.4 cua bao cao giai thich
# vi sao phai chon mot thu tu, va notebook 03 do xem thu tu co anh huong hay khong.
CHURN_SEQ = ["Geo_Germany", "Geo_Spain", "Gender_Male", "Age", "CreditScore",
             "EstimatedSalary", "Balance", "Tenure", "NumOfProducts", "HasCrCard",
             "IsActiveMember"]


def load_churn():
    return pd.read_csv(CHURN_CSV)


def churn_features(df):
    """Bo ba cot dinh danh, ma hoa hai cot chu. France va Female la muc goc."""
    X = pd.DataFrame({
        "Geo_Germany": (df["Geography"] == "Germany").astype(float),
        "Geo_Spain": (df["Geography"] == "Spain").astype(float),
        "Gender_Male": (df["Gender"] == "Male").astype(float),
        "Age": df["Age"].astype(float),
        "CreditScore": df["CreditScore"].astype(float),
        "EstimatedSalary": df["EstimatedSalary"].astype(float),
        "Balance": df["Balance"].astype(float),
        "Tenure": df["Tenure"].astype(float),
        "NumOfProducts": df["NumOfProducts"].astype(float),
        "HasCrCard": df["HasCrCard"].astype(float),
        "IsActiveMember": df["IsActiveMember"].astype(float),
    })
    return X[CHURN_SEQ], df["Exited"].to_numpy().astype(np.float32)


def churn_splits(order=None, seed=SEED):
    """Chia 64/16/20 co phan tang theo nhan, chuan hoa bang thong ke cua tap train.

    order: danh sach ten dac trung theo thu tu RNN se doc (mac dinh CHURN_SEQ).
    Tra ve X_* dang (N, T) - notebook tu them truc dac trung thanh (N, T, 1).
    """
    df = load_churn()
    X, y = churn_features(df)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.20, stratify=y[tr], random_state=SEED)
    mu = X.iloc[tr].mean()
    sd = X.iloc[tr].std(ddof=0)
    Z = (X - mu) / sd
    cols = list(order) if order is not None else list(CHURN_SEQ)
    Z = Z[cols].to_numpy().astype(np.float32)
    return dict(X_train=Z[tr], y_train=y[tr], X_val=Z[va], y_val=y[va],
                X_test=Z[te], y_test=y[te], features=cols,
                idx_train=tr, idx_val=va, idx_test=te)


def churn_logreg(S):
    """Baseline tuyen tinh: hoi quy logistic tren cung 11 dac trung da chuan hoa."""
    m = LogisticRegression(max_iter=1000)
    m.fit(S["X_train"], S["y_train"])
    return m.predict_proba(S["X_val"])[:, 1], m.predict_proba(S["X_test"])[:, 1]


def best_threshold(y, p):
    """Nguong xac suat cho F1 lon nhat tren tap validation (khong dung tap test)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def cls_metrics(y, p, thr):
    """Do do phan lop tren tap test. AUC khong phu thuoc nguong, cac do do con lai
    tinh voi nguong thr chon tu validation. Accuracy@0.5 ghi them de doi chieu."""
    yhat = (p >= thr).astype(int)
    return dict(
        roc_auc=round(float(roc_auc_score(y, p)), 4),
        pr_auc=round(float(average_precision_score(y, p)), 4),
        threshold=thr,
        accuracy=round(float(accuracy_score(y, yhat)), 4),
        precision=round(float(precision_score(y, yhat, zero_division=0)), 4),
        recall=round(float(recall_score(y, yhat, zero_division=0)), 4),
        f1=round(float(f1_score(y, yhat, zero_division=0)), 4),
        accuracy_at_05=round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
        confusion=confusion_matrix(y, yhat).tolist(),
    )


def permutation_importance(predict_fn, X, y, features, n_repeats=5, seed=SEED):
    """Muc giam ROC-AUC khi xao tron mot vi tri (mot dac trung) giua cac khach hang.

    predict_fn nhan mang (N, T) va tra ve xac suat (N,). Xao tron pha vo quan he giua
    dac trung do va nhan nhung giu nguyen phan phoi cua no.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict_fn(X))
    rows = []
    for j, name in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - roc_auc_score(y, predict_fn(Xp)))
        rows.append((name, float(np.mean(drops)), float(np.std(drops))))
    out = pd.DataFrame(rows, columns=["feature", "auc_drop", "std"])
    return out.sort_values("auc_drop", ascending=False).reset_index(drop=True)


# ====================================================================== stock

AMZN_CSV = DATA / "AMZN.csv"
TRAIN_END = "2018-12-31"          # train: 1997-2018
VAL_END = "2020-12-31"            # validation: 2019-2020, test: 2021-2023
WINDOW = 60                       # so ngay giao dich trong mot cua so dau vao
STOCK_FEATURES = ["ret", "range", "relvol"]


def load_amzn():
    df = pd.read_csv(AMZN_CSV, parse_dates=["Date"]).set_index("Date")
    return df


def stock_frame(df=None):
    """Ba dac trung theo ngay (deu khong phu thuoc muc gia nen dung duoc cho ca
    1997 lan 2023) va hai muc tieu du bao.

    ret     loi suat log ngay t: ln(C_t / C_{t-1})
    range   bien do trong ngay: ln(H_t / L_t), do bien dong
    relvol  khoi luong tuong doi: ln(V_t) tru trung binh ln(V) 20 ngay truoc do
    y_ret   loi suat ngay t+1                         (bai toan 1: huong va do lon)
    y_vol   bien dong thuc 5 ngay toi: sqrt(mean(ret_{t+1..t+5}^2))  (bai toan 2)
    """
    df = load_amzn() if df is None else df
    c = df["Adj Close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["ret"] = np.log(c / c.shift(1))
    f["range"] = np.log(df["High"] / df["Low"])
    lv = np.log(df["Volume"].astype(float))
    f["relvol"] = lv - lv.shift(1).rolling(20).mean()
    f["y_ret"] = f["ret"].shift(-1)
    r2 = f["ret"] ** 2
    f["y_vol"] = np.sqrt(sum(r2.shift(-k) for k in range(1, 6)) / 5)
    return f


def stock_splits(target="y_ret", window=WINDOW, features=STOCK_FEATURES, f=None):
    """Cat chuoi thanh cac cua so (N, window, F) va chia theo thoi gian.

    Mau thu i dung cac ngay t-window+1..t lam dau vao. Mau thuoc tap nao duoc quyet
    dinh bang ngay CUOI CUNG ma muc tieu cham toi (t+1 hoac t+5), nen khong mau train
    nao nhin thay du lieu cua giai doan validation/test. Thong ke chuan hoa chi tinh
    tren cac ngay thuoc giai doan train.
    """
    f = stock_frame() if f is None else f
    horizon = 1 if target == "y_ret" else 5
    valid = f[features].notna().all(axis=1).to_numpy()
    first = int(np.argmax(valid))                  # ngay dau co du ba dac trung
    dates = f.index
    train_mask_rows = (dates <= TRAIN_END) & valid
    mu = f.loc[train_mask_rows, features].mean()
    sd = f.loc[train_mask_rows, features].std(ddof=0)
    F = ((f[features] - mu) / sd).to_numpy().astype(np.float32)
    y_raw = f[target].to_numpy()

    ends = np.arange(first + window - 1, len(f) - horizon)   # ngay t cuoi cua so
    ends = ends[~np.isnan(y_raw[ends])]
    X = np.stack([F[t - window + 1:t + 1] for t in ends])
    last_target_day = dates[ends + horizon]
    split = np.where(last_target_day <= pd.Timestamp(TRAIN_END), "train",
                     np.where(last_target_day <= pd.Timestamp(VAL_END), "val", "test"))
    y = y_raw[ends]
    y_mu, y_sd = float(y[split == "train"].mean()), float(y[split == "train"].std())
    ys = ((y - y_mu) / y_sd).astype(np.float32)

    out = dict(features=list(features), target=target, window=window,
               y_mu=y_mu, y_sd=y_sd, x_mu=mu.to_dict(), x_sd=sd.to_dict())
    close = f["close"].to_numpy()
    for s in ("train", "val", "test"):
        m = split == s
        out["X_" + s] = X[m]
        out["y_" + s] = ys[m]                     # muc tieu da chuan hoa (de huan luyen)
        out["yraw_" + s] = y[m]                   # muc tieu that (de danh gia)
        out["t_" + s] = ends[m]                   # chi so ngay t trong f
        out["date_" + s] = dates[ends[m] + 1]     # ngay duoc du bao (t+1)
        out["close_t_" + s] = close[ends[m]]
        out["close_next_" + s] = close[ends[m] + 1]
    return out


def last_window(S, f=None):
    """Cua so 60 ngay cuoi cung cua du lieu (ket thuc 05/12/2023), chuan hoa bang
    dung thong ke cua tap train. Dung de du bao cho phien giao dich ke tiep."""
    f = stock_frame() if f is None else f
    mu, sd = pd.Series(S["x_mu"]), pd.Series(S["x_sd"])
    F = ((f[S["features"]] - mu) / sd).to_numpy().astype(np.float32)
    return F[-S["window"]:][None], f.index[-1], float(f["close"].iloc[-1])


def unscale(S, y_scaled):
    return np.asarray(y_scaled, dtype=np.float64).ravel() * S["y_sd"] + S["y_mu"]


def reg_metrics(S, pred, split="test"):
    """Do do cho hai bai toan. pred la muc tieu THAT (da unscale)."""
    y = S["yraw_" + split]
    pred = np.asarray(pred, dtype=np.float64).ravel()
    err = pred - y
    m = dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
             r2=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
             corr=float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else 0.0)
    if S["target"] == "y_ret":
        # R2 ngoai mau so voi du bao "loi suat bang 0" (gia ngay mai = gia hom nay)
        m["r2_vs_zero"] = float(1 - np.sum(err ** 2) / np.sum(y ** 2))
        # ti le doan dung chieu tang/giam; du bao dung bang 0 thi khong co chieu nao
        m["dir_acc"] = (float(np.mean((pred > 0) == (y > 0)))
                        if np.any(pred != 0) else float("nan"))
        c0, c1 = S["close_t_" + split], S["close_next_" + split]
        p_price = c0 * np.exp(pred)
        m["price_rmse"] = float(np.sqrt(np.mean((p_price - c1) ** 2)))
        m["price_mae"] = float(np.mean(np.abs(p_price - c1)))
        m["price_mape"] = float(np.mean(np.abs(p_price - c1) / c1) * 100)
    return {k: round(v, 6) for k, v in m.items()}


def stock_baselines(S):
    """Cac du bao khong can mang no-ron, dung lam moc so sanh cho moi mo hinh RNN."""
    out = {}
    n = len(S["yraw_test"])
    if S["target"] == "y_ret":
        out["Naive (loi suat 0)"] = np.zeros(n)
        out["Trung binh train"] = np.full(n, S["y_mu"])
        up = float(np.mean(S["yraw_train"] > 0))
        out["_up_share_test"] = float(np.mean(S["yraw_test"] > 0))
        out["_up_share_train"] = up
    else:
        # bien dong thuc 20 ngay gan nhat: du bao "ngay mai giong thang vua qua"
        ret = S["X_test"][:, :, S["features"].index("ret")] * S["x_sd"]["ret"] \
            + S["x_mu"]["ret"]
        out["Bien dong 20 ngay qua"] = np.sqrt(np.mean(ret[:, -20:] ** 2, axis=1))
        out["Trung binh train"] = np.full(n, S["y_mu"])
    # Ridge: mo hinh tuyen tinh tren dung cua so dau vao ma RNN nhin thay
    best = None
    for a in (1, 10, 100, 1000, 10000):
        r = Ridge(alpha=a).fit(S["X_train"].reshape(len(S["X_train"]), -1), S["y_train"])
        pv = unscale(S, r.predict(S["X_val"].reshape(len(S["X_val"]), -1)))
        e = np.sqrt(np.mean((pv - S["yraw_val"]) ** 2))
        if best is None or e < best[0]:
            best = (e, a, r)
    out["Ridge (cua so %d ngay)" % S["window"]] = unscale(
        S, best[2].predict(S["X_test"].reshape(n, -1)))
    out["_ridge_alpha"] = best[1]
    return out


def price_level_splits(window=WINDOW):
    """Cach lam hay gap trong cac bai huong dan: dua thang gia dong cua (chuan hoa
    min-max theo tap train) vao RNN va du bao gia ngay mai. Dung o notebook 03 de
    chi ra vi sao cach nay cho do thi dep nhung khong du bao duoc gi."""
    f = stock_frame()
    c = f["close"].to_numpy()
    dates = f.index
    tr_rows = dates <= TRAIN_END
    lo, hi = float(c[tr_rows].min()), float(c[tr_rows].max())
    z = ((c - lo) / (hi - lo)).astype(np.float32)
    ends = np.arange(window - 1, len(c) - 1)
    X = np.stack([z[t - window + 1:t + 1] for t in ends])[:, :, None]
    y = z[ends + 1]
    split = np.where(dates[ends + 1] <= pd.Timestamp(TRAIN_END), "train",
                     np.where(dates[ends + 1] <= pd.Timestamp(VAL_END), "val", "test"))
    out = dict(lo=lo, hi=hi)
    for s in ("train", "val", "test"):
        m = split == s
        out["X_" + s], out["y_" + s] = X[m], y[m]
        out["date_" + s] = dates[ends[m] + 1]
        out["close_t_" + s] = c[ends[m]]
        out["close_next_" + s] = c[ends[m] + 1]
    return out


def lag_profile(pred, actual, max_lag=5):
    """RMSE giua du bao ngay t va gia THAT cua ngay t-k, k = 0..max_lag.

    Neu RMSE nho nhat o k = 1 thi mo hinh chu yeu dang chep lai gia hom qua.
    """
    pred, actual = np.asarray(pred), np.asarray(actual)
    rows = []
    for k in range(max_lag + 1):
        a = actual[:len(actual) - k] if k else actual
        p = pred[k:]
        rows.append((k, float(np.sqrt(np.mean((p - a) ** 2)))))
    return pd.DataFrame(rows, columns=["lag_k", "rmse"])


def acf(x, nlags=30):
    """Ham tu tuong quan (tu viet vi may khong co statsmodels)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    d = np.sum(x ** 2)
    return np.array([1.0] + [np.sum(x[k:] * x[:-k]) / d for k in range(1, nlags + 1)])


# ====================================================================== ket qua

def save_result(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path = RESULTS / (name + ".json")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv),
                    encoding="utf-8")
    return path


def load_results(prefix=""):
    out = {}
    for p in sorted(RESULTS.glob(prefix + "*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0
