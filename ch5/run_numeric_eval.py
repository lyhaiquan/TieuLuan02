# -*- coding: utf-8 -*-
"""So sánh hai cách trả lời câu hỏi số trên bảng giá.

Cách A (bài 07): biến bảng thành câu thống kê theo năm, rồi truy hồi văn bản
và rút con số ra khỏi câu tìm được.

Cách B (chương này): định tuyến câu hỏi số sang bộ tính chạy thẳng trên bảng.

Tập câu hỏi được SINH TỰ ĐỘNG từ chính bảng giá nên đáp án đúng có sẵn, không
phải gán nhãn tay. Điều đó cho cỡ mẫu hàng trăm câu thay vì mười câu, nhưng
cũng giới hạn ý nghĩa: nó đo bộ định tuyến và bộ tính, không đo chất lượng
truy hồi tri thức. Phần đánh giá truy hồi nằm ở run_claims.py và phần diễn giải.

Chạy:  python run_numeric_eval.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import fin_core as F

OUT = os.path.join(F.HERE, "results")
os.makedirs(OUT, exist_ok=True)

ASSET_VI = {"GOLD": "vàng", "SILVER": "bạc", "OIL": "dầu", "SP500": "S&P 500",
            "BTC": "bitcoin", "USD": "đồng đô", "VIX": "VIX", "US10Y": "lợi suất trái phiếu"}

# Sai số chấp nhận được khi so đáp án. Câu thống kê của cách A làm tròn tới
# 0,1% nên ngưỡng phải rộng hơn mức làm tròn đó, nếu không là phạt cách A vì
# chuyện làm tròn chứ không phải vì trả lời sai.
TOL = {"return": 0.005, "volatility": 0.005, "drawdown": 0.005,
       "correlation": 0.02, "last": None}


def make_evalset(prices):
    """Sinh câu hỏi số kèm đáp án đúng tính thẳng từ bảng."""
    rows = []
    assets = ["GOLD", "SILVER", "OIL", "SP500", "BTC"]
    years = range(2020, 2026)

    for a in assets:
        for y in years:
            s, e = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)
            for metric, templ in [
                ("return", "Lợi suất của %s năm %d là bao nhiêu?"),
                ("volatility", "Biến động của %s năm %d là bao nhiêu?"),
                ("drawdown", "Mức sụt sâu nhất của %s năm %d là bao nhiêu?"),
            ]:
                gold = F.compute(prices, metric, a, s, e)
                if gold is not None:
                    rows.append({"q": templ % (ASSET_VI[a], y), "metric": metric,
                                 "gold": gold, "scope": "năm", "asset": a})
            for qt in range(1, 5):
                qs = pd.Timestamp(y, 3 * qt - 2, 1)
                qe = qs + pd.offsets.QuarterEnd(0)
                for metric, templ in [
                    ("return", "Lợi suất của %s quý %d/%d là bao nhiêu?"),
                    ("volatility", "Biến động của %s quý %d/%d là bao nhiêu?"),
                ]:
                    gold = F.compute(prices, metric, a, qs, qe)
                    if gold is not None:
                        rows.append({"q": templ % (ASSET_VI[a], qt, y),
                                     "metric": metric, "gold": gold,
                                     "scope": "quý", "asset": a})

    for y in years:                       # câu hỏi tương quan, chỉ cách B làm được
        s, e = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)
        for a, b in [("GOLD", "USD"), ("GOLD", "US10Y"), ("GOLD", "SP500"),
                     ("SILVER", "GOLD"), ("OIL", "SP500")]:
            gold = F.compute(prices, "correlation", a, s, e, other=b)
            if gold is not None:
                rows.append({"q": "Tương quan giữa %s và %s năm %d là bao nhiêu?"
                                  % (ASSET_VI[a], ASSET_VI[b], y),
                             "metric": "correlation", "gold": gold,
                             "scope": "năm", "asset": a})
    return pd.DataFrame(rows)


class ExtractiveBaseline:
    """Cách A: truy hồi câu thống kê rồi rút số ra bằng biểu thức chính quy."""

    PAT = {"return": r"price return ([+-]?[\d.]+)%",
           "volatility": r"annualized volatility ([\d.]+)%",
           "drawdown": r"maximum drawdown ([+-]?[\d.]+)%",
           "last": r"last price ([\d.]+)"}

    def __init__(self, docs):
        self.docs = docs
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform([d["text"] for d in docs])

    def answer(self, q, metric):
        # Câu hỏi tiếng Việt, kho câu tiếng Anh: thêm tên tài sản và năm vào
        # truy vấn để truy hồi có chỗ bám, đây là ưu ái cho cách A chứ không
        # phải bất lợi.
        probe = q
        for en, vi in ASSET_VI.items():
            if vi.lower() in q.lower():
                probe += " " + en
        s = (self.mat @ self.vec.transform([probe]).T).toarray().ravel()
        hit = self.docs[int(np.argmax(s))]
        pat = self.PAT.get(metric)
        if pat is None:
            return None, hit["source"]
        m = re.search(pat, hit["text"])
        return (float(m.group(1)) / 100.0 if m else None), hit["source"]


def correct(pred, gold, metric):
    if pred is None or gold is None:
        return False
    tol = TOL[metric]
    return abs(pred - gold) <= tol


def main():
    prices = F.load_prices()
    ev = make_evalset(prices)
    base = ExtractiveBaseline(F.make_stats_corpus(prices))

    recs = []
    for _, r in ev.iterrows():
        pa = F.parse_question(r["q"])
        b_pred, b_src = base.answer(r["q"], r["metric"])
        out = F.answer_numeric(prices, pa)
        recs.append({
            **r.to_dict(),
            "A_pred": b_pred, "A_src": b_src, "A_ok": correct(b_pred, r["gold"], r["metric"]),
            "B_pred": out.get("value"), "B_ok": correct(out.get("value"), r["gold"], r["metric"]),
            "B_kind": pa["kind"], "B_metric": pa["metric"],
        })
    df = pd.DataFrame(recs)

    summary = {"so_cau_hoi": len(df)}
    for name, sub in [("tong", df), ("theo_nam", df[df["scope"] == "năm"]),
                      ("theo_quy", df[df["scope"] == "quý"])]:
        summary[name] = {
            "n": len(sub),
            "A_truy_hoi_cau_thong_ke": round(float(sub["A_ok"].mean()), 4),
            "B_dinh_tuyen_sang_bo_tinh": round(float(sub["B_ok"].mean()), 4),
        }
    summary["theo_do_do"] = {
        m: {"n": int(len(s)),
            "A": round(float(s["A_ok"].mean()), 4),
            "B": round(float(s["B_ok"].mean()), 4)}
        for m, s in df.groupby("metric")}

    with open(os.path.join(OUT, "numeric_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    df.to_csv(os.path.join(OUT, "numeric_eval.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
