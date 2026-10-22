# -*- coding: utf-8 -*-
"""Đánh giá truy hồi tin tức trên hai trục độc lập nhau.

Trục 1 - rò rỉ nhìn trước: tài liệu trả về có nằm trong cửa sổ thời gian của
câu hỏi không. Đo bằng ngày đăng.

Trục 2 - độ liên quan: tài liệu trả về có đúng nội dung câu hỏi không. Đo bằng
cách so hướng giá nêu trong tiêu đề với hướng mà câu hỏi hỏi; câu hỏi "vì sao
vàng giảm" mà trả về tin vàng tăng là trượt.

Phải có cả hai, vì một bộ lọc thời gian hoàn hảo vẫn có thể trả về tài liệu
đúng tháng nhưng sai nội dung, và chỉ đo trục 1 thì không thấy.

Chạy:  python run_narrative_eval.py
"""
import json
import os

import pandas as pd

import fin_core as F

OUT = os.path.join(F.HERE, "results")
os.makedirs(OUT, exist_ok=True)

# Câu hỏi có hướng để đo được độ liên quan, kèm hướng đúng mong đợi.
DIRECTED = [("vì sao giá vàng giảm tháng %d/%d", -1),
            ("vì sao giá vàng tăng tháng %d/%d", +1)]
PLAIN = ["tin tức về vàng tháng %d/%d"]


def make_questions(news, min_docs=5):
    """Chỉ sinh câu hỏi cho tháng có đủ tin; hỏi tháng không có tin là đo lỗ
    hổng dữ liệu chứ không đo hệ thống."""
    d = pd.to_datetime(news["date"], utc=True)
    counts = d.dt.to_period("M").value_counts()
    qs = []
    for period, n in sorted(counts.items()):
        if n < min_docs:
            continue
        w = (pd.Timestamp(period.start_time, tz="UTC"),
             pd.Timestamp(period.end_time, tz="UTC"))
        for t, dirn in DIRECTED:
            qs.append({"q": t % (period.month, period.year), "window": w,
                       "dir": dirn, "thang": str(period), "n_thang": int(n)})
        for t in PLAIN:
            qs.append({"q": t % (period.month, period.year), "window": w,
                       "dir": 0, "thang": str(period), "n_thang": int(n)})
    return qs


def relevance(hits, want_dir):
    """Tỉ lệ tiêu đề trả về có hướng giá khớp với hướng câu hỏi hỏi."""
    if want_dir == 0 or not hits:
        return None
    hit_dirs = [F._direction(h["title"]) for h in hits]
    return sum(1 for d in hit_dirs if d == want_dir) / len(hits)


def main(k=5):
    news = F.load_news()
    ret = F.TimeScopedRetriever(news)
    qs = make_questions(news)

    rows = []
    for item in qs:
        start, end = item["window"]
        for tf in (True, False):
            for ex in (True, False):
                hits = ret.retrieve(item["q"], item["window"], k=k,
                                    time_filter=tf, expand=ex)
                rows.append({
                    "q": item["q"], "thang": item["thang"], "dir": item["dir"],
                    "loc_thoi_gian": tf, "mo_rong_truy_van": ex,
                    "so_tai_lieu": len(hits),
                    "tu_tuong_lai": sum(h["date"] > end for h in hits),
                    "ngoai_cua_so": sum((h["date"] > end) or (h["date"] < start)
                                        for h in hits),
                    "do_lien_quan": relevance(hits, item["dir"]),
                })
    df = pd.DataFrame(rows)

    summary = {"so_cau_hoi": len(qs), "so_thang": df["thang"].nunique(), "k": k,
               "che_do_truy_hoi": ret.mode, "cau_hinh": {}}
    for tf in (True, False):
        for ex in (True, False):
            sub = df[(df["loc_thoi_gian"] == tf) & (df["mo_rong_truy_van"] == ex)]
            rel = sub["do_lien_quan"].dropna()
            summary["cau_hinh"]["loc=%s, mo_rong=%s" % (tf, ex)] = {
                "ty_le_tu_tuong_lai": round(float(sub["tu_tuong_lai"].sum()
                                                  / max(1, sub["so_tai_lieu"].sum())), 4),
                "ty_le_ngoai_cua_so": round(float(sub["ngoai_cua_so"].sum()
                                                  / max(1, sub["so_tai_lieu"].sum())), 4),
                "do_lien_quan_trung_binh": round(float(rel.mean()), 4) if len(rel) else None,
                "so_cau_do_duoc_lien_quan": int(len(rel)),
            }

    with open(os.path.join(OUT, "narrative_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    df.to_csv(os.path.join(OUT, "narrative_eval.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
