# -*- coding: utf-8 -*-
"""Tien xu ly dung chung cho Chuong 2: lap lai DUNG quy trinh cua Assignment 03
(lam sach, ma hoa, chia tap phan tang seed 42, chuan hoa fit tren Train) de ba cai dat
(NumPy / PyTorch / Keras) va cac mo hinh scikit-learn doc cung mot bo X, y.
Ket qua ghi ra data/ch2_<he thong>.npz."""
import re
import numpy as np
import pandas as pd
import dl_scratch as dl

SEED = 42
CLASSES = ["negative", "neutral", "positive"]


def diabetes():
    df = pd.read_csv("data/diabetes_prediction_dataset.csv")
    df = df.drop_duplicates()
    df = df[df["gender"] != "Other"]
    df = df[df["age"] >= 1.0]
    df = df[(df["bmi"] >= 12.0) & (df["bmi"] <= 80.0)]
    GENDER = {"Female": 0, "Male": 1}
    SMOKE = {"No Info": 0, "never": 1, "ever": 2, "not current": 3, "former": 4, "current": 5}
    FEATS = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi",
             "HbA1c_level", "blood_glucose_level"]
    enc = df.copy()
    enc["gender"] = enc["gender"].map(GENDER)
    enc["smoking_history"] = enc["smoking_history"].map(SMOKE)
    X = enc[FEATS].to_numpy(dtype=float)
    y = enc["diabetes"].to_numpy(dtype=float).reshape(-1, 1)
    tr, te = dl.stratified_split(y, test_size=0.2, seed=SEED)
    sc = dl.StandardScalerScratch().fit(X[tr])
    return dict(X_tr=sc.transform(X[tr]), X_te=sc.transform(X[te]), y_tr=y[tr], y_te=y[te],
                feats=np.array(FEATS), Xraw_te=X[te])


def _f(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"[\d.,]+", str(s))
    if not m:
        return np.nan
    t = m.group(0).replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def _rooms(s):
    if pd.isna(s):
        return np.nan
    if "nhiều hơn" in str(s).lower():
        return 10.0
    return _f(s)


class SmoothTargetEncoder:
    def __init__(self, m=20.0):
        self.m = m

    def fit(self, keys, y):
        self.global_ = float(np.mean(y))
        s = pd.DataFrame({"k": np.asarray(keys), "y": np.asarray(y)})
        agg = s.groupby("k")["y"].agg(["mean", "size"])
        self.map_ = ((agg["size"] * agg["mean"] + self.m * self.global_) / (agg["size"] + self.m)).to_dict()
        return self

    def transform(self, keys):
        return np.array([self.map_.get(k, self.global_) for k in np.asarray(keys)], dtype=float)


def housing():
    raw = pd.read_csv("data/hanoi_housing.csv")
    d = pd.DataFrame({
        "area": raw["Diện tích"].map(_f),
        "ppm2": raw["Giá/m2"].map(lambda s: _f(s) * (1000.0 if "tỷ" in str(s) else 1.0)),
        "beds": raw["Số phòng ngủ"].map(_rooms), "floors": raw["Số tầng"].map(_rooms),
        "quan": raw["Quận"], "phuong": raw["Huyện"], "loai": raw["Loại hình nhà ở"],
        "phaply": raw["Giấy tờ pháp lý"], "ngay": pd.to_datetime(raw["Ngày"], errors="coerce")})
    d["price"] = d["area"] * d["ppm2"] / 1000.0
    d = d.dropna(subset=["area", "ppm2", "price"])
    d = d[(d["area"] >= 10) & (d["area"] <= 1000)]
    d = d[(d["price"] >= 0.3) & (d["price"] <= 100)]
    d = d.drop_duplicates(subset=["area", "ppm2", "beds", "quan", "phuong", "loai"]).reset_index(drop=True)
    d["log_price"] = np.log(d["price"])
    tr, te = dl.stratified_split(d["quan"].astype("category").cat.codes.to_numpy(), test_size=0.2, seed=SEED)
    y_log = d["log_price"].to_numpy()
    beds_med = float(np.nanmedian(d["beds"].to_numpy()[tr])); floors_med = float(np.nanmedian(d["floors"].to_numpy()[tr]))
    beds = d["beds"].fillna(beds_med).clip(1, 10).to_numpy(float)
    floors_na = d["floors"].isna().to_numpy(float)
    floors = d["floors"].fillna(floors_med).clip(1, 10).to_numpy(float)
    te_q = SmoothTargetEncoder(m=25).fit(d["quan"].to_numpy()[tr], y_log[tr])
    te_p = SmoothTargetEncoder(m=15).fit(d["phuong"].to_numpy()[tr], y_log[tr])
    loai = pd.get_dummies(d["loai"].fillna("Khác"), prefix="lo").astype(float); loai = loai[loai.columns[:3]]
    phaply = pd.get_dummies(d["phaply"].fillna("Khác"), prefix="pl").astype(float); phaply = phaply[phaply.columns[:3]]
    X = np.column_stack([np.log(d["area"].to_numpy()), beds, floors, floors_na, d["area"].to_numpy() / beds,
                         beds * floors, te_q.transform(d["quan"].to_numpy()), te_p.transform(d["phuong"].to_numpy()),
                         d["ngay"].dt.month.fillna(6).to_numpy(float), loai.to_numpy(), phaply.to_numpy()])
    sc = dl.StandardScalerScratch().fit(X[tr])
    y_mu, y_sd = float(y_log[tr].mean()), float(y_log[tr].std())
    return dict(X_tr=sc.transform(X[tr]), X_te=sc.transform(X[te]),
                y_tr=((y_log[tr] - y_mu) / y_sd).reshape(-1, 1), y_te=((y_log[te] - y_mu) / y_sd).reshape(-1, 1),
                y_mu=np.array(y_mu), y_sd=np.array(y_sd), price_te=d["price"].to_numpy()[te])


def comments():
    from sklearn.feature_extraction.text import TfidfVectorizer
    d = pd.read_csv("data/youtube_comments.csv")
    d = d.dropna(subset=["Comment", "Sentiment"])
    d = d.drop_duplicates(subset=["Comment"])
    d["n_words"] = d["Comment"].str.split().str.len()
    d = d[d["n_words"] >= 3]
    d = d[d["Sentiment"].isin(CLASSES)].reset_index(drop=True)
    y = d["Sentiment"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    texts = d["Comment"].astype(str).to_numpy()
    tr, te = dl.stratified_split(y, test_size=0.2, seed=SEED)
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english", min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform(texts[tr]).toarray(); Xte = vec.transform(texts[te]).toarray()
    return dict(X_tr=Xtr.astype(np.float32), X_te=Xte.astype(np.float32), y_tr=y[tr], y_te=y[te])


if __name__ == "__main__":
    for name, fn in [("diabetes", diabetes), ("housing", housing), ("comments", comments)]:
        D = fn()
        np.savez_compressed("data/ch2_%s.npz" % name, **D)
        print(name, {k: v.shape for k, v in D.items()})
