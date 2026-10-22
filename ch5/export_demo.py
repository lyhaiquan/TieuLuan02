# -*- coding: utf-8 -*-
"""Xuất dữ liệu cho demo chạy trong trình duyệt của Chương 5.

Demo không gọi máy chủ nào: toàn bộ bảng giá, kho câu thống kê của cách A, và
kết quả kiểm chứng khẳng định được nhúng thẳng vào một tệp .js. Nhờ vậy phép
tính số thật sự chạy trong trình duyệt chứ không phải đáp án nấu sẵn.

Chạy:  python export_demo.py
"""
import json
import os

import numpy as np
import pandas as pd

import fin_core as F

DEMO = os.path.join(os.path.dirname(F.HERE), "demo", "data")
os.makedirs(DEMO, exist_ok=True)
OUT = os.path.join(DEMO, "ch5_data.js")

MONTHS = [("2026-%02d" % m) for m in range(1, 11)]


def main():
    prices = F.load_prices()
    news = F.load_news()

    # Bảng giá: làm tròn để tệp nhẹ mà không đổi kết quả ở 4 chữ số thập phân.
    px = prices.round(4)
    series = {c: [None if pd.isna(v) else float(v) for v in px[c]] for c in px.columns}
    dates = [d.strftime("%Y-%m-%d") for d in px.index]

    # Kho câu thống kê của cách A, để demo so được hai cách trên cùng câu hỏi.
    stats_docs = [d["text"] for d in F.make_stats_corpus(prices)]

    # Tương quan theo quý, dùng cho đồ thị và cho phần giải thích chế độ.
    corr = F.correlation_edges(prices, freq="QE")
    pairs = {}
    for a, b in [("GOLD", "USD"), ("GOLD", "US10Y"), ("GOLD", "SP500"),
                 ("GOLD", "OIL"), ("GOLD", "SILVER"), ("GOLD", "VIX")]:
        sub, _ = F.regime_shifts(corr, (a, b))
        whole, _ = F.metric_correlation(prices, a, b, prices.index[0], prices.index[-1])
        pairs["%s-%s" % (a, b)] = {
            "periods": sub["period"].tolist(),
            "corr": [float(x) for x in sub["corr"]],
            "n_days": [int(x) for x in sub["n_days"]],
            "whole": round(float(whole), 3),
        }

    # Câu trả lời nhân quả dựng sẵn cho từng tháng có tin, vì phần này cần mô
    # hình nhúng nên không chạy được trong trình duyệt.
    ret = F.TimeScopedRetriever(news)
    causal = {}
    for m in MONTHS:
        start = pd.Timestamp(m + "-01", tz="UTC")
        end = start + pd.offsets.MonthEnd(0)
        for direction, label in [("giảm", "down"), ("tăng", "up")]:
            q = "vì sao giá vàng %s tháng %d/%d" % (direction, start.month, start.year)
            a = F.answer(q, prices, ret, k=5)
            causal["%s|%s" % (m, label)] = {
                "cau_hoi": q,
                "bao_chi": a["bao_chi_quy_cho"],
                "khong_do_duoc": a["nguyen_nhan_khong_do_duoc_bang_gia"],
                "doi_chieu": [
                    {k: (None if (isinstance(d[k], float) and np.isnan(d[k])) else d[k])
                     for k in ("nguyen_nhan", "ket_luan", "corr", "tieu_de")}
                    for d in a["doi_chieu_du_lieu"]],
                "tai_lieu": [{"ngay": h["date"].strftime("%Y-%m-%d"),
                              "tieu_de": h["title"], "nguon": h["source"]}
                             for h in a["tai_lieu"]],
            }

    blob = {
        "dates": dates, "series": series,
        "assets": list(px.columns),
        "stats_docs": stats_docs,
        "pairs": pairs,
        "causal": causal,
        "aliases": {k: v["aliases"] for k, v in F.ENTITIES.items()},
        "rate_series": list(F.RATE_SERIES),
        "min_days": F.MIN_DAYS,
        "ket_qua": {
            "numeric": json.load(open(os.path.join(F.HERE, "results",
                                                   "numeric_eval.json"), encoding="utf-8")),
            "claims": json.load(open(os.path.join(F.HERE, "results",
                                                  "claims_gold.json"), encoding="utf-8")),
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.CH5 = ")
        json.dump(blob, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print("%s: %.0f KB" % (os.path.basename(OUT), os.path.getsize(OUT) / 1024))
    print("  %d phiên, %d câu thống kê, %d cặp, %d câu hỏi nhân quả"
          % (len(dates), len(stats_docs), len(pairs), len(causal)))


if __name__ == "__main__":
    main()
