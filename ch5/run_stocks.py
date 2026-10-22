# -*- coding: utf-8 -*-
"""Miền chứng khoán: cạnh belongs_to, cạnh tương quan theo quý, và phép đo cho
thấy vì sao bộ trích khẳng định nhân quả KHÔNG dùng được ở miền này.

Kết quả quan trọng nhất của script này là một con số âm: trên 574 tiêu đề tin
cổ phiếu chỉ trích được 2 khẳng định, so với 309 trên 627 tiêu đề tin vàng.
Lý do nằm ở thể loại tin: vàng là tài sản vĩ mô nên được viết theo khuôn
"giá X giảm vì Y tăng", còn tin về từng mã cổ phiếu phần lớn là tin doanh
nghiệp (bổ nhiệm, giải thưởng, kết quả kinh doanh) không nêu quan hệ giá.

Chạy:  python run_stocks.py
"""
import json
import os

import pandas as pd

import fin_core as F

OUT = os.path.join(F.HERE, "results")
os.makedirs(OUT, exist_ok=True)

PAIRS = [("VNM", "FPT"), ("VNM", "VCB"), ("FPT", "VCB"),
         ("HPG", "VIC"), ("VCB", "VIC")]


def main():
    # Mở rộng bảng thực thể sang các mã cổ phiếu, dùng chung bộ máy với vàng.
    F.ENTITIES.update({k: v for k, v in F.stock_entities().items()
                       if k in F.TICKERS})
    F.ALIAS_RX = F._alias_patterns()

    prices = F.load_prices("stock_prices.csv")
    news = F.load_news("stock_news.csv")
    info = F.load_company_info()

    be = F.belongs_to_edges(info)
    corr = F.correlation_edges(prices, freq="QE")

    pair_stats = {}
    for a, b in PAIRS:
        sub, flips = F.regime_shifts(corr, (a, b))
        whole, n = F.metric_correlation(prices, a, b, prices.index[0], prices.index[-1])
        pair_stats["%s-%s" % (a, b)] = {
            "corr_toan_giai_doan": round(whole, 3),
            "corr_min": float(sub["corr"].min()), "corr_max": float(sub["corr"].max()),
            "bien_do": round(float(sub["corr"].max() - sub["corr"].min()), 3),
            "so_lan_doi_dau": len(flips),
        }

    by_ticker = {}
    total_claims = 0
    for tic in F.TICKERS:
        cl = F.claims_from_news(news, asset=tic)
        by_ticker[tic] = len(cl)
        total_claims += len(cl)

    summary = {
        "so_ma": len(F.TICKERS),
        "so_phien": len(prices),
        "tin_sau_khu_trung_lap": len(news),
        "canh_belongs_to": len(be),
        "canh_correlates_with": len(corr),
        "so_lat_thoi_gian": int(corr["period"].nunique()),
        "theo_cap": pair_stats,
        "khang_dinh_trich_duoc": {"tong": total_claims, "theo_ma": by_ticker},
        "so_sanh_voi_mien_vang": {
            "vang_tin": 627, "vang_khang_dinh": 309,
            "co_phieu_tin": len(news), "co_phieu_khang_dinh": total_claims,
        },
    }
    with open(os.path.join(OUT, "stocks_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    be.to_csv(os.path.join(OUT, "stocks_belongs_to.csv"), index=False, encoding="utf-8-sig")
    corr.to_csv(os.path.join(OUT, "stocks_corr_edges.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
