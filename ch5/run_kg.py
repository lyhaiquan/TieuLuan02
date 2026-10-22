# -*- coding: utf-8 -*-
"""Dựng đồ thị tri thức có kiểu, có thời gian, rồi ghi số liệu ra results/.

Hai loại cạnh được giữ riêng:
  correlates_with  tính từ giá, cắt theo quý
  claims_cause     trích từ tin tức, gộp theo tháng

Chạy:  python run_kg.py
"""
import json
import os

import pandas as pd

import fin_core as F

OUT = os.path.join(F.HERE, "results")
os.makedirs(OUT, exist_ok=True)

PAIRS = [("GOLD", "USD"), ("GOLD", "US10Y"), ("GOLD", "SP500"),
         ("GOLD", "OIL"), ("GOLD", "SILVER"), ("GOLD", "VIX")]


def main():
    prices = F.load_prices()
    news = F.load_news()

    corr = F.correlation_edges(prices, freq="QE")
    claims = F.claims_from_news(news, asset="GOLD")
    verified = F.verify_all(prices, claims)
    cedges = F.claim_edges(verified)

    pair_stats, flip_rows = {}, []
    for a, b in PAIRS:
        sub, flips = F.regime_shifts(corr, (a, b))
        pair_stats["%s-%s" % (a, b)] = {
            "so_lat": len(sub),
            "corr_min": float(sub["corr"].min()),
            "corr_max": float(sub["corr"].max()),
            "corr_trung_binh": round(float(sub["corr"].mean()), 3),
            "bien_do": round(float(sub["corr"].max() - sub["corr"].min()), 3),
            "so_lan_doi_dau": len(flips),
        }
        for _, r in flips.iterrows():
            flip_rows.append({"cap": "%s-%s" % (a, b), **r.to_dict()})

    # Một con số duy nhất cho cả giai đoạn, để đối chiếu với cách cắt lát.
    whole = {}
    for a, b in PAIRS:
        c, n = F.metric_correlation(prices, a, b, prices.index[0], prices.index[-1])
        whole["%s-%s" % (a, b)] = {"corr_toan_giai_doan": round(c, 3), "n_days": n}

    summary = {
        "canh_correlates_with": len(corr),
        "so_lat_thoi_gian": int(corr["period"].nunique()),
        "canh_claims_cause": len(cedges),
        "so_nut": len(F.ENTITIES),
        "nut_co_chuoi_gia": sum(1 for e in F.ENTITIES if F.series_of(e)),
        "theo_cap": pair_stats,
        "mot_canh_tinh_cho_ca_giai_doan": whole,
    }
    with open(os.path.join(OUT, "kg_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    corr.to_csv(os.path.join(OUT, "kg_corr_edges.csv"), index=False, encoding="utf-8-sig")
    if len(cedges):
        cedges.to_csv(os.path.join(OUT, "kg_claim_edges.csv"), index=False,
                      encoding="utf-8-sig")
    pd.DataFrame(flip_rows).to_csv(os.path.join(OUT, "kg_regime_shifts.csv"),
                                   index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
