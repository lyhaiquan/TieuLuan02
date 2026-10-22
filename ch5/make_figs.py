# -*- coding: utf-8 -*-
"""Vẽ bốn hình cho chương 5, lấy số liệu từ results/*.json và *.csv.

Chạy:  python make_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

import fin_core as F                     # noqa: E402

RES = os.path.join(F.HERE, "results")
FIGS = os.path.join(os.path.dirname(F.HERE), "figs")
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def fig_regime():
    """Tương quan theo quý so với một hệ số duy nhất cho cả giai đoạn."""
    edges = pd.read_csv(os.path.join(RES, "kg_corr_edges.csv"))
    kg = load("kg_summary.json")
    pairs = [("GOLD", "USD"), ("GOLD", "SP500"), ("GOLD", "SILVER")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    for ax, (a, b) in zip(axes, pairs):
        sub = edges[((edges.src == a) & (edges.dst == b)) |
                    ((edges.src == b) & (edges.dst == a))].sort_values("period")
        x = range(len(sub))
        ax.bar(x, sub["corr"], color=["#c0392b" if v < 0 else "#2471a3"
                                      for v in sub["corr"]])
        whole = kg["mot_canh_tinh_cho_ca_giai_doan"]["%s-%s" % (a, b)]["corr_toan_giai_doan"]
        ax.axhline(whole, color="black", ls="--", lw=1.2,
                   label="một cạnh cho cả giai đoạn = %.2f" % whole)
        ax.axhline(0, color="gray", lw=.6)
        ax.set_title("%s – %s" % (a, b))
        ax.set_xticks([0, len(sub) // 2, len(sub) - 1])
        ax.set_xticklabels([sub.iloc[0]["period"][:7], sub.iloc[len(sub) // 2]["period"][:7],
                            sub.iloc[-1]["period"][:7]], fontsize=7)
        ax.legend(fontsize=6.5, loc="lower left")
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("tương quan theo quý")
    fig.suptitle("Cạnh tĩnh che mất hiện tượng: ba cặp, cùng cách tính", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ch5_regime.png"), bbox_inches="tight")
    plt.close(fig)


def fig_numeric():
    """So sánh hai cách trả lời câu hỏi số."""
    d = load("numeric_eval.json")
    groups = [("Theo năm", d["theo_nam"]), ("Theo quý", d["theo_quy"]),
              ("Tổng", d["tong"])]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    x = range(len(groups))
    w = 0.36
    a = [g[1]["A_truy_hoi_cau_thong_ke"] for g in groups]
    b = [g[1]["B_dinh_tuyen_sang_bo_tinh"] for g in groups]
    ax.bar([i - w / 2 for i in x], a, w, label="A: bảng thành câu rồi truy hồi",
           color="#95a5a6")
    ax.bar([i + w / 2 for i in x], b, w, label="B: định tuyến sang bộ tính",
           color="#2471a3")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va + .02, "%.2f" % va, ha="center", fontsize=8)
        ax.text(i + w / 2, vb + .02, "%.2f" % vb, ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["%s (n=%d)" % (g[0], g[1]["n"]) for g in groups])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("tỉ lệ trả lời đúng")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=.3)
    ax.set_title("Câu hỏi số: %d câu sinh tự động từ bảng giá" % d["so_cau_hoi"],
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ch5_numeric.png"), bbox_inches="tight")
    plt.close(fig)


def fig_retrieval():
    """Rò rỉ nhìn trước và độ liên quan, bốn cấu hình."""
    d = load("narrative_eval.json")["cau_hinh"]
    labels, leak, rel = [], [], []
    name = {"loc=True, mo_rong=True": "lọc + mở rộng",
            "loc=True, mo_rong=False": "lọc, không mở rộng",
            "loc=False, mo_rong=True": "không lọc, mở rộng",
            "loc=False, mo_rong=False": "không lọc, không mở rộng"}
    for k, v in d.items():
        labels.append(name.get(k, k))
        leak.append(v["ty_le_tu_tuong_lai"])
        rel.append(v["do_lien_quan_trung_binh"] or 0.0)
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = range(len(labels))
    w = 0.36
    ax.bar([i - w / 2 for i in x], leak, w, label="tài liệu đến từ tương lai",
           color="#c0392b")
    ax.bar([i + w / 2 for i in x], rel, w, label="độ liên quan (hướng khớp)",
           color="#27ae60")
    for i, (a, b) in enumerate(zip(leak, rel)):
        ax.text(i - w / 2, a + .015, "%.2f" % a, ha="center", fontsize=8)
        ax.text(i + w / 2, b + .015, "%.2f" % b, ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=.3)
    ax.set_title("Truy hồi tin tức: hai trục phải đo cùng lúc", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ch5_retrieval.png"), bbox_inches="tight")
    plt.close(fig)


def fig_claims():
    """Kết luận kiểm chứng và lý do không kiểm chứng được."""
    d = load("claims_gold.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.2))

    ly_do = d["ly_do_khong_kiem_chung_duoc"]
    keys = ["nguyên nhân\nkhông có\nchuỗi giá", "nhiều thực thể\ntrong một vế",
            "không rõ hướng\ncủa một vế", "câu nêu\nnghịch lý"]
    vals = list(ly_do.values())
    a1.barh(range(len(vals))[::-1], vals, color="#7f8c8d")
    a1.set_yticks(range(len(vals))[::-1])
    a1.set_yticklabels(keys[:len(vals)], fontsize=7.5)
    for i, v in enumerate(vals):
        a1.text(v + 3, len(vals) - 1 - i, str(v), va="center", fontsize=8)
    a1.set_xlim(0, max(vals) * 1.25)
    a1.set_title("%d/%d khẳng định không kiểm chứng được"
                 % (d["khang_dinh"] - d["kiem_chung_duoc"], d["khang_dinh"]),
                 fontsize=9.5)
    a1.grid(axis="x", alpha=.3)

    real = d["ty_le_nhat_quan_ngay_that"]
    null = d["kiem_dinh_vo_hieu_ngay_ngau_nhien"]
    a2.bar([0, 1], [real, null["trung_binh"]], 0.5,
           yerr=[0, null["do_lech_chuan"]], capsize=5,
           color=["#2471a3", "#95a5a6"])
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["ngày thật\n(n=%d)" % d["kiem_chung_duoc"],
                        "ngày ngẫu nhiên\n(%d lần lặp)" % null["so_lan_lap"]],
                       fontsize=8)
    a2.set_ylim(0, 1.1)
    a2.set_ylabel("tỉ lệ nhất quán")
    a2.text(0, real + .04, "%.2f" % real, ha="center", fontsize=8)
    a2.text(1, null["trung_binh"] + null["do_lech_chuan"] + .04,
            "%.2f" % null["trung_binh"], ha="center", fontsize=8)
    a2.set_title("Kiểm định vô hiệu: phần lớn độ nhất quán\nđến từ chế độ thị trường",
                 fontsize=9.5)
    a2.grid(axis="y", alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ch5_claims.png"), bbox_inches="tight")
    plt.close(fig)


def fig_graph():
    """Ba loai canh cua do thi: correlates_with, claims_cause, belongs_to."""
    import math
    corr = pd.read_csv(os.path.join(RES, "kg_corr_edges.csv"))
    clm = pd.read_csv(os.path.join(RES, "kg_claim_edges.csv"))
    bt = pd.read_csv(os.path.join(RES, "stocks_belongs_to.csv"))
    quy, thang = "2026-06-30", "2026-06"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.0),
                                 gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- Panel trai: mien vang, mot lat thoi gian ----
    ngoai = ["SILVER", "SP500", "BTC", "OIL", "US10Y", "USD", "VIX"]
    goc = {n: math.radians(135 - i * 270.0 / (len(ngoai) - 1))
           for i, n in enumerate(ngoai)}
    pos = {n: (1.0 * math.cos(g), 1.0 * math.sin(g)) for n, g in goc.items()}
    pos["GOLD"] = (0.0, 0.0)

    sub = corr[(corr.period == quy) & ((corr.src == "GOLD") | (corr.dst == "GOLD"))]
    for _, e in sub.iterrows():
        kia = e.dst if e.src == "GOLD" else e.src
        (x0, y0), (x1, y1) = pos["GOLD"], pos[kia]
        mau = "#c0392b" if e["corr"] < 0 else "#2471a3"
        a1.plot([x0, x1], [y0, y1], color=mau, lw=0.8 + 3.4 * abs(e["corr"]),
                alpha=.85, zorder=1)
        a1.text(x0 + .60 * (x1 - x0), y0 + .60 * (y1 - y0), "%+.2f" % e["corr"],
                fontsize=7.4, color=mau, ha="center", va="center", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.8))

    # canh claims_cause cua chinh thang do
    cl = clm[clm.period == thang]
    ngoai_gia = {"FED": (-1.85, 0.62), "GEOPOLITICS": (-1.85, 0.0),
                 "INFLATION": (-1.85, -0.62)}
    for _, e in cl.iterrows():
        if e.src in ngoai_gia:
            x0, y0 = ngoai_gia[e.src]
            a1.annotate("", xy=(-0.31, 0.0), xytext=(x0 + .42, y0),
                        arrowprops=dict(arrowstyle="-|>", ls="--", lw=1.2,
                                        color="#7f8c8d", shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=%.2f"
                                        % (-0.16 if y0 > 0 else (0.16 if y0 < 0 else 0))))
            a1.text(x0 + .62, y0 + (.17 if y0 > 0 else (-.23 if y0 < 0 else .13)),
                    "%d tin" % e.n_claims,
                    fontsize=7.2, color="#7f8c8d", ha="center", zorder=5)
            a1.add_patch(plt.Rectangle((x0 - .42, y0 - .17), .84, .34, fc="#ecf0f1",
                                       ec="#7f8c8d", ls="--", lw=1.1, zorder=3))
            a1.text(x0, y0, e.src, fontsize=7.2, ha="center", va="center", zorder=4)
        elif e.src in pos:            # nguyen nhan co chuoi gia: doi chieu duoc
            x1, y1 = pos[e.src]
            a1.annotate("", xy=(0.04, -0.33), xytext=(x1 * .84, y1 * .84 - .12),
                        arrowprops=dict(arrowstyle="-|>", ls="--", lw=1.3,
                                        color="#1e8449", shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=0.5"))
            a1.text(1.02, -1.30, "%d tin,\nđối chiếu được" % e.n_claims,
                    fontsize=7.2, color="#1e8449", ha="center", zorder=5)

    for n, (x, y) in pos.items():
        r = .30 if n == "GOLD" else .25
        a1.add_patch(plt.Circle((x, y), r, fc="#f6d55c" if n == "GOLD" else "white",
                                ec="#34495e", lw=1.3, zorder=3))
        a1.text(x, y, n, fontsize=7.4 if len(n) > 5 else 8.2, ha="center",
                va="center", zorder=4,
                fontweight="bold" if n == "GOLD" else "normal")

    a1.set_xlim(-2.45, 1.55)
    a1.set_ylim(-1.95, 1.45)
    a1.set_aspect("equal")
    a1.axis("off")
    a1.set_title("Miền vàng, lát quý 2/2026: cạnh tính từ giá (liền)\n"
                 "và cạnh trích từ tin (nét đứt)", fontsize=9.5)
    chu = [plt.Line2D([], [], color="#2471a3", lw=2.6, label="correlates_with, dương"),
           plt.Line2D([], [], color="#c0392b", lw=2.6, label="correlates_with, âm"),
           plt.Line2D([], [], color="#1e8449", lw=1.6, ls="--",
                      label="claims_cause, kiểm chứng được"),
           plt.Line2D([], [], color="#7f8c8d", lw=1.6, ls="--",
                      label="claims_cause, không có chuỗi giá")]
    a1.legend(handles=chu, fontsize=7, loc="upper center", frameon=False,
              ncol=2, bbox_to_anchor=(0.5, 0.055), columnspacing=1.4)

    # ---- Panel phai: belongs_to ----
    bt2 = bt[bt.src.isin(["FPT", "VCB"])]
    p2 = {"FPT": (0, 1.05), "VCB": (0, -1.05), "Vietnam": (1.05, 0.0),
          "Information Technology Services": (2.25, 1.5),
          "Technology": (2.25, 0.72),
          "Banks - Regional": (2.25, -0.72),
          "Financial Services": (2.25, -1.5)}
    for _, e in bt2.iterrows():
        x0, y0 = p2[e.src]
        x1, y1 = p2[e.dst]
        a2.annotate("", xy=(x1 - .34, y1), xytext=(x0 + .30, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#34495e",
                                    shrinkA=0, shrinkB=0))
        a2.text((x0 + x1) / 2 + .06, (y0 + y1) / 2 + .10, e.chi_tiet,
                fontsize=6.8, color="#34495e", ha="center")
    for n, (x, y) in p2.items():
        ma = n in ("FPT", "VCB")
        w = .30 if ma or n == "Vietnam" else 1.15
        a2.add_patch(plt.Rectangle((x - w / 2 - .02, y - .17), w + .04, .34,
                                   fc="#eaf2f8" if ma else "white",
                                   ec="#34495e", lw=1.2, zorder=3))
        a2.text(x, y, n if len(n) < 20 else n.replace(" ", "\n", 1),
                fontsize=6.6 if len(n) > 14 else 7.8, ha="center", va="center",
                zorder=4, fontweight="bold" if ma else "normal")
    a2.set_xlim(-0.45, 3.05)
    a2.set_ylim(-1.95, 1.95)
    a2.axis("off")
    a2.set_title("Miền chứng khoán: cạnh belongs_to tra cứu được\n"
                 "(vẽ 2 trong 5 mã, mỗi mã 3 cạnh)", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ch5_graph.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_graph()
    fig_regime()
    fig_numeric()
    fig_retrieval()
    fig_claims()
    for f in sorted(os.listdir(FIGS)):
        if f.startswith("ch5_"):
            print("%-22s %6.0f KB" % (f, os.path.getsize(os.path.join(FIGS, f)) / 1024))
