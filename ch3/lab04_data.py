"""
lab04_data.py - Chuan bi du lieu dung chung cho Exercise 04 (CNN 1D).

Nguyen tac: nhung gi phai giong nhau giua ba cai dat:
    Input data / Train-test split / Preprocessing / Model idea / Loss / Metrics
    phai GIONG NHAU giua 3 cai dat (NumPy - PyTorch - TensorFlow),
    chi co "implementation layer" la thay doi.
=> Toan bo tien xu ly duoc dat trong file nay va duoc ca 3 notebook import.

3 bo du lieu theo yeu cau cua de bai:
    1. Diabetes    - tabular  -> phan lop nhi phan   (sigmoid, BCE)
    2. House Price - tabular  -> hoi quy             (linear, MSE)
    3. Comments    - text     -> phan lop 3 lop      (softmax, CE)
"""

import os
import re
import numpy as np
import pandas as pd

# Ba file CSV nam trong folder data/ cua project. Neu folder do bi thieu (vi du chi
# copy rieng file .py di noi khac) thi quay ve thu muc C:\DATA noi CSV duoc tai ve.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = r"C:\DATA"

SEED = 42

# Kich thuoc mau: giam Diabetes xuong 20k de notebook "from scratch" (NumPy)
# chay trong thoi gian hop ly; ca 3 framework dung CUNG mot mau -> so sanh cong bang.
DIABETES_SAMPLE = 20000
TEXT_MAX_LEN = 60          # do dai chuoi token sau khi pad/truncate
TEXT_VOCAB_SIZE = 15000    # so tu trong tu dien (ke ca <pad>, <unk>)
HOUSING_TOP_PROVINCES = 8  # so tinh/thanh duoc ma hoa one-hot rieng, con lai gop "other"


# ----------------------------------------------------------------------
# Tien ich chung
# ----------------------------------------------------------------------
def _standardize(X_train, X_test):
    """Chuan hoa z-score, thong ke chi tinh tren tap train (tranh data leakage)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def _split(X, y, test_size=0.2, stratify=None, seed=SEED):
    """Train/test split thu cong bang NumPy (khong phu thuoc framework nao)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    if stratify is None:
        rng.shuffle(idx)
        n_test = int(round(n * test_size))
        te, tr = idx[:n_test], idx[n_test:]
    else:
        te_parts, tr_parts = [], []
        for c in np.unique(stratify):
            ci = idx[stratify == c]
            rng.shuffle(ci)
            k = int(round(len(ci) * test_size))
            te_parts.append(ci[:k])
            tr_parts.append(ci[k:])
        te = np.concatenate(te_parts)
        tr = np.concatenate(tr_parts)
        rng.shuffle(te)
        rng.shuffle(tr)
    return X[tr], X[te], y[tr], y[te]


# ----------------------------------------------------------------------
# 1. DIABETES  (tabular -> binary classification)
# ----------------------------------------------------------------------
def load_diabetes():
    """
    diabetes_prediction_dataset.csv: 100.000 benh nhan, 8 dac trung + nhan 'diabetes'.
    8 dac trung, tuc X in R^8, duoc coi nhu mot chuoi 1D do dai 8.
    """
    path = os.path.join(DATA_DIR, "diabetes_prediction_dataset.csv")
    df = pd.read_csv(path)

    # Ma hoa bien hang muc
    df = df[df["gender"] != "Other"].copy()          # 18/100000 dong, bo cho gon
    df["gender"] = (df["gender"] == "Male").astype(float)
    smoke_map = {"No Info": 0, "never": 1, "former": 2, "not current": 3,
                 "ever": 4, "current": 5}
    df["smoking_history"] = df["smoking_history"].map(smoke_map).astype(float)

    feat_names = ["gender", "age", "hypertension", "heart_disease",
                  "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
    X = df[feat_names].to_numpy(dtype=np.float64)
    y = df["diabetes"].to_numpy(dtype=np.float64)

    # Lay mau phan tang 20.000 dong
    rng = np.random.default_rng(SEED)
    keep = []
    for c in (0, 1):
        ci = np.where(y == c)[0]
        k = int(round(DIABETES_SAMPLE * len(ci) / len(y)))
        keep.append(rng.choice(ci, size=k, replace=False))
    keep = np.concatenate(keep)
    X, y = X[keep], y[keep]

    Xtr, Xte, ytr, yte = _split(X, y, 0.2, stratify=y)
    Xtr, Xte, _, _ = _standardize(Xtr, Xte)
    return dict(name="Diabetes", task="binary", X_train=Xtr, X_test=Xte,
                y_train=ytr, y_test=yte, feature_names=feat_names,
                n_features=len(feat_names))


# ----------------------------------------------------------------------
# 2. HOUSE PRICE (tabular -> regression)
# ----------------------------------------------------------------------
def load_housing():
    """
    vietnam_housing_dataset.csv: 30.229 tin ban nha, Price tinh bang ty VND.
    Muc tieu hoi quy: y = log1p(Price). 19 dac trung (6 so + 4 nhi phan
    + 9 one-hot tinh/thanh) -> chuoi 1D do dai 19.
    """
    path = os.path.join(DATA_DIR, "vietnam_housing_dataset.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")

    df = df[(df["Price"] > 0) & df["Price"].notna() & df["Area"].notna()].copy()
    df = df[df["Area"].between(10, 1000)]            # loai ngoai lai dien tich

    num_cols = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
    for c in num_cols:                                # thieu nhieu -> dien median
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df["legal_ok"] = (df["Legal status"] == "Have certificate").astype(float)
    df["furn_full"] = (df["Furniture state"] == "Full").astype(float)
    df["furn_basic"] = (df["Furniture state"] == "Basic").astype(float)
    df["has_direction"] = df["House direction"].notna().astype(float)

    # Vi tri la yeu to quyet dinh gia nha: tach tinh/thanh tu cot Address
    # va ma hoa one-hot cho HOUSING_TOP_PROVINCES tinh pho bien nhat.
    prov = (df["Address"].astype(str).str.split(",").str[-1]
            .str.strip().str.rstrip(".").str.strip())
    top = prov.value_counts().index[:HOUSING_TOP_PROVINCES]
    prov = prov.where(prov.isin(top), "other")
    prov_cols = []
    for p in list(top) + ["other"]:
        col = "prov_" + (p.replace(" ", "_").lower())
        df[col] = (prov == p).astype(float)
        prov_cols.append(col)

    feat_names = (num_cols + ["legal_ok", "furn_full", "furn_basic", "has_direction"]
                  + prov_cols)
    X = df[feat_names].to_numpy(dtype=np.float64)
    price = df["Price"].to_numpy(dtype=np.float64)    # ty VND
    y = np.log1p(price)                               # hoi quy tren log-price

    Xtr, Xte, ytr, yte = _split(X, y, 0.2)
    Xtr, Xte, _, _ = _standardize(Xtr, Xte)

    # Chuan hoa y de huan luyen on dinh; luu mu/sd de doi nguoc ve ty VND khi danh gia
    y_mu, y_sd = ytr.mean(), ytr.std()
    return dict(name="House Price", task="regression", X_train=Xtr, X_test=Xte,
                y_train=(ytr - y_mu) / y_sd, y_test=(yte - y_mu) / y_sd,
                y_train_log=ytr, y_test_log=yte, y_mu=y_mu, y_sd=y_sd,
                feature_names=feat_names, n_features=len(feat_names))


def housing_to_billion(y_scaled, meta):
    """Doi output da chuan hoa -> gia thuc (ty VND)."""
    y_log = y_scaled * meta["y_sd"] + meta["y_mu"]
    return np.expm1(y_log)


# ----------------------------------------------------------------------
# 3. COMMENTS (text -> 3-class classification)
# ----------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z]+")


def _tokenize(text):
    return _TOKEN_RE.findall(str(text).lower())


def load_comments():
    """
    YoutubeCommentsDataSet.csv: 18.408 comment, nhan negative/neutral/positive.
    Text la du lieu TU NHIEN nhat cho Conv1D: cua so k token bat cac cum tu
    dia phuong nhu [very, good] hoac [not, good].
    """
    path = os.path.join(DATA_DIR, "YoutubeCommentsDataSet.csv")
    df = pd.read_csv(path).dropna(subset=["Comment", "Sentiment"])

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y = df["Sentiment"].map(label_map).to_numpy(dtype=np.int64)
    texts = df["Comment"].to_numpy(dtype=object)

    tr_t, te_t, ytr, yte = _split(texts, y, 0.2, stratify=y)

    # Tu dien chi xay tren TAP TRAIN
    from collections import Counter
    cnt = Counter()
    tok_tr = [_tokenize(t) for t in tr_t]
    for toks in tok_tr:
        cnt.update(toks)
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in cnt.most_common(TEXT_VOCAB_SIZE - 2):
        vocab[w] = len(vocab)

    def encode(tok_list):
        out = np.zeros((len(tok_list), TEXT_MAX_LEN), dtype=np.int64)
        for i, toks in enumerate(tok_list):
            ids = [vocab.get(w, 1) for w in toks[:TEXT_MAX_LEN]]
            out[i, :len(ids)] = ids
        return out

    Xtr = encode(tok_tr)
    Xte = encode([_tokenize(t) for t in te_t])
    return dict(name="Comments", task="multiclass", X_train=Xtr, X_test=Xte,
                y_train=ytr, y_test=yte, vocab=vocab, vocab_size=len(vocab),
                max_len=TEXT_MAX_LEN, n_classes=3,
                class_names=["negative", "neutral", "positive"])


# ----------------------------------------------------------------------
# Metrics dung chung cho ca 3 notebook
# ----------------------------------------------------------------------
def binary_metrics(y_true, p_hat, thr=0.5):
    y_pred = (p_hat >= thr).astype(int)
    y_true = y_true.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


def multiclass_metrics(y_true, probs):
    y_pred = probs.argmax(axis=1)
    K = probs.shape[1]
    acc = float((y_pred == y_true).mean())
    per = []
    for c in range(K):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per.append(dict(precision=p, recall=r, f1=f, support=int((y_true == c).sum())))
    macro_f1 = float(np.mean([d["f1"] for d in per]))
    w = np.array([d["support"] for d in per], dtype=float)
    f1s = np.array([d["f1"] for d in per], dtype=float)
    weighted_f1 = float(np.sum(f1s * w) / w.sum())
    return dict(accuracy=acc, macro_f1=macro_f1, weighted_f1=weighted_f1, per_class=per)


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return dict(mae=mae, rmse=rmse, r2=r2)


def save_result(framework, dataset, metrics, extra=None, out_dir="results"):
    """Ghi ket qua ra JSON de notebook bao cao tong hop lai."""
    import json
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(framework=framework, dataset=dataset, metrics=metrics)
    if extra:
        rec.update(extra)
    fn = os.path.join(out_dir, f"{framework}_{dataset}.json".replace(" ", "_").lower())
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False, default=float)
    return fn


if __name__ == "__main__":
    for loader in (load_diabetes, load_housing, load_comments):
        d = loader()
        print(d["name"], d["task"], d["X_train"].shape, d["X_test"].shape,
              "y:", np.unique(d["y_train"])[:4] if d["task"] != "regression"
              else round(float(d["y_train"].mean()), 3))
