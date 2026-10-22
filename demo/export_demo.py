# -*- coding: utf-8 -*-
"""Huan luyen va xuat trong so cho cac demo (chay trong trinh duyet, khong can server).
python export_demo.py ch2|ch3|ch4  ->  demo/data/<chuong>_model.js  (window.MODEL = {...})"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "data"
OUT.mkdir(parents=True, exist_ok=True)
which = sys.argv[1]


def dump(name, obj):
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    (OUT / name).write_text("window.MODEL=" + s + ";", encoding="utf-8")
    print("wrote", name, "%.0f KB" % (len(s) / 1024))


def r(a, nd=5):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


if which == "ch2":
    # Chuong 2: MLP 8-32-16-8-1 viet bang NumPy thuan (dl_scratch), huan luyen nhu ch2_run, seed 42
    os.chdir(ROOT / "ch2"); sys.path.insert(0, str(ROOT / "ch2"))
    import pandas as pd
    import dl_scratch as dl
    import ch2_data as C
    D = C.diabetes()
    Xte_raw, yte = D["Xraw_te"], D["y_te"].ravel()
    # trung binh/do lech chuan tren Train: suy tu X_tr chuan hoa va du lieu tho -> tinh lai tu CSV theo dung quy trinh
    df = pd.read_csv("data/diabetes_prediction_dataset.csv").drop_duplicates()
    df = df[df["gender"] != "Other"]; df = df[df["age"] >= 1.0]; df = df[(df["bmi"] >= 12.0) & (df["bmi"] <= 80.0)]
    enc = df.copy(); enc["gender"] = enc["gender"].map({"Female": 0, "Male": 1})
    enc["smoking_history"] = enc["smoking_history"].map({"No Info": 0, "never": 1, "ever": 2, "not current": 3, "former": 4, "current": 5})
    X = enc[list(D["feats"])].to_numpy(float); y = enc["diabetes"].to_numpy(float).reshape(-1, 1)
    tr, te = dl.stratified_split(y, 0.2, 42)
    sc = dl.StandardScalerScratch().fit(X[tr])
    assert np.allclose(sc.transform(X[te]), D["X_te"])
    m = dl.MLPScratch([8, 32, 16, 8, 1], "binary", lr=0.1, epochs=1500, seed=42)
    t = time.time(); m.fit(D["X_tr"], D["y_tr"]); print("train %.0fs" % (time.time() - t))
    p = m.predict_proba(D["X_te"]).ravel()
    mt = dl.binary_metrics(yte, p, 0.5)
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [dl.binary_metrics(yte, p, g)["f1"] for g in grid]
    # vai mau kiem tra de doi chieu voi JS
    idx = [0, 1, 2, 3, 4, 5, 6, 7]
    dump("ch2_model.js", dict(
        feats=list(D["feats"]), mu=r(sc.mean_), sd=r(sc.std_), layers=[dict(W=r(W), b=r(b.ravel())) for W, b in m.params],
        test=dict(acc=mt["accuracy"], precision=mt["precision"], recall=mt["recall"], f1=mt["f1"], best_thr=float(grid[int(np.argmax(f1s))])),
        check=[dict(x=r(Xte_raw[i], 4), p=float(p[i])) for i in idx],
        pos_examples=[r(Xte_raw[i], 4) for i in np.where(yte == 1)[0][:3]], neg_examples=[r(Xte_raw[i], 4) for i in np.where(yte == 0)[0][:3]]))

elif which == "ch3":
    # Chuong 3: TextCNN (Embedding-Conv1D-Pool-Conv1D-GlobalMax-Dense) huan luyen bang PyTorch, suy luan bang JS
    os.chdir(ROOT / "ch3"); sys.path.insert(0, str(ROOT / "ch3"))
    import torch, torch.nn as nn
    import lab04_data as L
    com = L.load_comments()

    class TextCNN(nn.Module):
        def __init__(s, V, emb=32, ch=64, n=3, p=0.3):
            super().__init__(); s.emb = nn.Embedding(V, emb, padding_idx=0); s.conv1 = nn.Conv1d(emb, ch, 3); s.pool = nn.MaxPool1d(2); s.conv2 = nn.Conv1d(ch, ch, 3); s.drop = nn.Dropout(p); s.fc = nn.Linear(ch, n)

        def forward(s, i):
            h = s.emb(i).transpose(1, 2); h = torch.relu(s.conv1(h)); h = s.pool(h); h = torch.relu(s.conv2(h)); h = h.amax(2); return s.fc(s.drop(h))
    torch.manual_seed(42)
    net = TextCNN(com["vocab_size"])
    for mod in net.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu"); nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, 0, .05)
            with torch.no_grad(): mod.weight[0].zero_()
    Xtr, ytr = torch.tensor(com["X_train"]), torch.tensor(com["y_train"]); Xte, yte = torch.tensor(com["X_test"]), com["y_test"]
    opt = torch.optim.Adam(net.parameters(), 1.5e-3); g = torch.Generator().manual_seed(42)
    for ep in range(4):
        net.train(); perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 64):
            b = perm[i:i + 64]; opt.zero_grad(); nn.functional.cross_entropy(net(Xtr[b]), ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        P = torch.softmax(net(Xte), 1).numpy()
    acc = float((P.argmax(1) == yte).mean()); print("acc", acc)
    inv = [None] * com["vocab_size"]
    for w, i in com["vocab"].items(): inv[i] = w
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    texts = ["I love this song, it is amazing and beautiful", "this is the worst video ever, I hate it", "the weather in the video was cloudy today", "thank you so much, great work", "what a terrible and boring waste of time"]
    chk = []
    for tx in texts:
        ids = [com["vocab"].get(w, 1) for w in L._tokenize(tx)[:60]]; ids += [0] * (60 - len(ids))
        with torch.no_grad(): chk.append(dict(text=tx, p=torch.softmax(net(torch.tensor([ids])), 1)[0].tolist()))
    dump("ch3_model.js", dict(vocab=inv, emb=r(sd["emb.weight"], 3), c1w=r(sd["conv1.weight"], 4), c1b=r(sd["conv1.bias"], 4), c2w=r(sd["conv2.weight"], 4), c2b=r(sd["conv2.bias"], 4),
                              fw=r(sd["fc.weight"], 4), fb=r(sd["fc.bias"], 4), classes=com["class_names"], test_acc=acc, maxlen=60, check=chk))

elif which == "ch4":
    # Chuong 4: GRU (NumPy thuan, tu viet BPTT) du bao bien dong 5 phien toi cua AMZN, suy luan bang JS
    os.chdir(ROOT / "ch4"); sys.path.insert(0, str(ROOT / "ch4"))
    import lab06_data as D
    import rnn_numpy as R
    S = D.stock_splits("y_vol"); hp = D.STOCK_HP
    m = R.SeqNetNP("GRU", len(S["features"]), hp["hidden"], hp["dropout"], 42)
    info = R.fit(m, S, R.mse_grad, hp, 42)
    pred = D.unscale(S, m.forward(S["X_test"].astype(np.float64)))
    mt = D.reg_metrics(S, pred); print(mt, info["epochs"])
    P = m.params()
    base = D.stock_baselines(S)
    dump("ch4_model.js", dict(H=hp["hidden"], F=len(S["features"]), feats=S["features"], Wih=r(P["rnn.W_ih"]), Whh=r(P["rnn.W_hh"]), bih=r(P["rnn.b_ih"]), bhh=r(P["rnn.b_hh"]),
                              Wo=r(P["W_o"]), bo=r(P["b_o"]), y_mu=S["y_mu"], y_sd=S["y_sd"], metrics=mt,
                              X=[r(x, 3) for x in S["X_test"]], y=r(S["yraw_test"], 5), pred=r(pred, 5), base20=r(base["Bien dong 20 ngay qua"], 5),
                              dates=[str(d)[:10] for d in S["date_test"]], close=r(S["close_t_test"], 2)))
