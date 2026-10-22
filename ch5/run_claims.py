# -*- coding: utf-8 -*-
"""Trích khẳng định nhân quả từ tin tức rồi kiểm chứng bằng dữ liệu giá.

Ghi kết quả ra results/claims_*.json để chương 5 của tiểu luận đọc lại, không
nhập tay con số nào.

Chạy:  python run_claims.py
"""
import json
import os

import pandas as pd

import fin_core as F

OUT = os.path.join(F.HERE, "results")
os.makedirs(OUT, exist_ok=True)


def run(news_file, asset, tag):
    prices = F.load_prices("gold_market.csv")
    news = F.load_news(news_file)
    raw = pd.read_csv(os.path.join(F.DATA, news_file))
    claims = F.claims_from_news(news, asset=asset)
    ver = F.verify_all(prices, claims)

    checkable = ver[ver["verdict"] != "không kiểm chứng được"]
    summary = {
        "tin_goc": len(raw),
        "tin_sau_khu_trung_lap": len(news),
        "khang_dinh": len(ver),
        "kiem_chung_duoc": len(checkable),
        "theo_ket_luan": ver["verdict"].value_counts().to_dict(),
        "ly_do_khong_kiem_chung_duoc":
            ver[ver["verdict"] == "không kiểm chứng được"]["reason"]
            .value_counts().to_dict(),
        "theo_nguyen_nhan": claims["driver"].value_counts().to_dict(),
        "ty_le_nguyen_nhan_khong_co_chuoi_gia": round(
            float((~claims["driver"].map(lambda d: F.series_of(d) is not None)).mean()), 4),
    }
    summary["kiem_dinh_vo_hieu_ngay_ngau_nhien"] = shuffle_test(prices, claims)
    summary["ty_le_nhat_quan_ngay_that"] = round(float(
        (checkable["verdict"] == "nhất quán").mean()), 4) if len(checkable) else None

    # Xuất mẫu ngẫu nhiên để soi tay độ chính xác của bộ trích, lấy trên MỌI
    # kết luận chứ không chỉ các ca mâu thuẫn: ba vòng sửa trước chỉ soi phần
    # bất đồng nên cả một lớp lỗi chưa từng được nhìn tới.
    if len(checkable):
        checkable.sample(n=min(30, len(checkable)), random_state=42)            .to_csv(os.path.join(OUT, "claims_%s_audit.csv" % tag), index=False,
                    encoding="utf-8-sig")

    with open(os.path.join(OUT, "claims_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    ver.to_csv(os.path.join(OUT, "claims_%s.csv" % tag), index=False,
               encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary



def shuffle_test(prices, claims, n_rep=20, seed=42):
    """Kiểm định vô hiệu: gán cho mỗi khẳng định một ngày ngẫu nhiên rồi kiểm
    chứng lại.

    Cần thiết vì tương quan GOLD-USD âm ở gần hết các quý. Nếu gán ngày bừa mà
    tỉ lệ "nhất quán" vẫn cao như khi gán đúng ngày, thì bộ kiểm chứng đang xác
    nhận dấu của chế độ thị trường nói chung, không xác nhận khẳng định cụ thể
    của bản tin đó.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(claims["date"], utc=True)
    lo, hi = dates.min().value, dates.max().value
    rates = []
    for _ in range(n_rep):
        fake = claims.copy()
        fake["date"] = pd.to_datetime(rng.integers(lo, hi, len(fake)), utc=True)
        v = F.verify_all(prices, fake)
        ok = v[v["verdict"] != "không kiểm chứng được"]
        rates.append(float((ok["verdict"] == "nhất quán").mean()) if len(ok) else 0.0)
    return {"trung_binh": round(float(np.mean(rates)), 4),
            "do_lech_chuan": round(float(np.std(rates)), 4),
            "so_lan_lap": n_rep}


if __name__ == "__main__":
    run("gold_news.csv", "GOLD", "gold")
