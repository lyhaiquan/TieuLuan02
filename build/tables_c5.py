# -*- coding: utf-8 -*-
"""Bang va so lieu cho Chuong 5 (Knowledge Graph va RAG cho mien tai chinh).

Moi con so doc tu ch5/results/*.json, khong nhap tay.
"""
from tables_base import J, f, pc, n, loc


def C5(name):
    return J("ch5/results/%s" % name)


def t_c5_data():
    c = C5("claims_gold.json")
    s = C5("stocks_summary.json")
    return ("Dữ liệu hai miền tài chính của Chương 5",
            ["", "Vàng", "Chứng khoán"],
            [["Bảng số", "giá vàng và 7 yếu tố vĩ mô (USD, lợi suất TP Mỹ 10 năm, bạc, dầu, S&P 500, VIX, Bitcoin)",
              "giá đóng cửa 5 mã HOSE: VNM, FPT, VCB, HPG, VIC"],
             ["Số phiên", "2.480 (2020-01-01 đến 2026-10-15)", "%s" % n(s["so_phien"])],
             ["Tin tức gốc", "%s tiêu đề" % n(c["tin_goc"]), "%s tiêu đề" % n(s["tin_sau_khu_trung_lap"])],
             ["Sau khử trùng lặp", "%s" % n(c["tin_sau_khu_trung_lap"]), "%s" % n(s["tin_sau_khu_trung_lap"])],
             ["Khẳng định nhân quả trích được", "%s" % n(c["khang_dinh"]),
              "%s" % n(s["khang_dinh_trich_duoc"]["tong"])],
             ["Nguồn", "yfinance, Google News RSS", "yfinance, Google News RSS"]])


def t_c5_edges():
    kg = C5("kg_summary.json")
    s = C5("stocks_summary.json")
    return ("Ba loại cạnh của đồ thị, tách theo nguồn gốc",
            ["Loại cạnh", "Đến từ đâu", "Độ tin cậy", "Số cạnh"],
            [["`belongs_to`", "dữ liệu tham chiếu (mã → ngành → quốc gia)",
              "tra cứu được, coi như đúng", "%s" % n(s["canh_belongs_to"])],
             ["`correlates_with`", "tính từ giá, cắt theo quý",
              "đo được, kèm số phiên", "%s (vàng) + %s (cổ phiếu)"
              % (n(kg["canh_correlates_with"]), n(s["canh_correlates_with"]))],
             ["`claims_cause`", "trích từ tiêu đề tin tức",
              "là lời người viết, phải kiểm chứng", "%s" % n(kg["canh_claims_cause"])]])


def t_c5_regime():
    kg = C5("kg_summary.json")
    rows = []
    for pair, v in kg["theo_cap"].items():
        w = kg["mot_canh_tinh_cho_ca_giai_doan"][pair]["corr_toan_giai_doan"]
        rows.append([pair.replace("-", " – "), f(w, 3), f(v["corr_min"], 3),
                     f(v["corr_max"], 3), f(v["bien_do"], 3), str(v["so_lan_doi_dau"])])
    return ("Một cạnh tĩnh so với cạnh cắt theo quý (27 quý)",
            ["Cặp", "Một cạnh cho cả giai đoạn", "Nhỏ nhất", "Lớn nhất", "Biên độ", "Số lần đổi dấu"],
            rows)


def t_c5_numeric():
    d = C5("numeric_eval.json")
    rows = []
    for key, lab in [("theo_nam", "Theo năm"), ("theo_quy", "Theo quý"), ("tong", "Tổng")]:
        v = d[key]
        rows.append([lab, n(v["n"]), f(v["A_truy_hoi_cau_thong_ke"], 3),
                     f(v["B_dinh_tuyen_sang_bo_tinh"], 3)])
    for m, v in sorted(d["theo_do_do"].items()):
        rows.append(["– riêng %s" % m, n(v["n"]), f(v["A"], 3), f(v["B"], 3)])
    return ("Câu hỏi số: truy hồi câu thống kê so với định tuyến sang bộ tính",
            ["Nhóm câu hỏi", "Số câu", "A: bảng thành câu", "B: định tuyến"],
            rows)


def t_c5_retrieval():
    d = C5("narrative_eval.json")["cau_hinh"]
    name = {"loc=True, mo_rong=True": "Có lọc, có mở rộng",
            "loc=True, mo_rong=False": "Có lọc, không mở rộng",
            "loc=False, mo_rong=True": "Không lọc, có mở rộng",
            "loc=False, mo_rong=False": "Không lọc, không mở rộng"}
    rows = []
    for k, v in d.items():
        rel = v["do_lien_quan_trung_binh"]
        rows.append([name.get(k, k), pc(v["ty_le_tu_tuong_lai"]),
                     pc(v["ty_le_ngoai_cua_so"]),
                     f(rel, 2) if rel is not None else "–"])
    return ("Truy hồi tin tức: rò rỉ nhìn trước và độ liên quan",
            ["Cấu hình", "Tài liệu từ tương lai", "Ngoài cửa sổ", "Độ liên quan"],
            rows)


def t_c5_claims():
    d = C5("claims_gold.json")
    rows = [["Tổng khẳng định trích được", n(d["khang_dinh"]), ""],
            ["Kiểm chứng được bằng dữ liệu giá", n(d["kiem_chung_duoc"]),
             pc(d["kiem_chung_duoc"] / d["khang_dinh"])]]
    for k, v in d["theo_ket_luan"].items():
        rows.append(["– %s" % k, n(v), pc(v / d["khang_dinh"])])
    for k, v in d["ly_do_khong_kiem_chung_duoc"].items():
        rows.append(["Không kiểm chứng được: %s" % k, n(v), pc(v / d["khang_dinh"])])
    return ("Kết quả kiểm chứng khẳng định nhân quả (miền vàng)",
            ["", "Số lượng", "Tỉ lệ"], rows)


def t_c5_null():
    d = C5("claims_gold.json")
    nu = d["kiem_dinh_vo_hieu_ngay_ngau_nhien"]
    return ("Kiểm định vô hiệu cho bộ kiểm chứng",
            ["", "Tỉ lệ nhất quán", "Ghi chú"],
            [["Ngày thật của bản tin", f(d["ty_le_nhat_quan_ngay_that"], 4),
              "n = %s khẳng định" % n(d["kiem_chung_duoc"])],
             ["Ngày gán ngẫu nhiên", "%s ± %s" % (f(nu["trung_binh"], 4), f(nu["do_lech_chuan"], 4)),
              "%d lần lặp" % nu["so_lan_lap"]],
             ["Chênh lệch", f(d["ty_le_nhat_quan_ngay_that"] - nu["trung_binh"], 4),
              "khoảng %s độ lệch chuẩn" % f((d["ty_le_nhat_quan_ngay_that"] - nu["trung_binh"]) / nu["do_lech_chuan"], 1)]])


def t_c5_stocks():
    s = C5("stocks_summary.json")
    c = s["so_sanh_voi_mien_vang"]
    return ("Bộ trích khẳng định phụ thuộc thể loại tin, không phụ thuộc miền dữ liệu",
            ["", "Vàng", "Chứng khoán"],
            [["Số tiêu đề", n(c["vang_tin"]), n(c["co_phieu_tin"])],
             ["Khẳng định nhân quả trích được", n(c["vang_khang_dinh"]), n(c["co_phieu_khang_dinh"])],
             ["Tỉ lệ", pc(c["vang_khang_dinh"] / c["vang_tin"]),
              pc(c["co_phieu_khang_dinh"] / c["co_phieu_tin"])],
             ["Thể loại tiêu đề chủ yếu", "tin thị trường theo khuôn “X giảm vì Y tăng”",
              "tin doanh nghiệp: bổ nhiệm, giải thưởng, kết quả kinh doanh"]])


def t_c5_diabetes():
    """Ket qua mien y van lay lai tu bai thuc hanh truoc (lab 07)."""
    d = C5("lab07_summary.json")["eval"]
    rows = []
    for k in ("2", "4", "6"):
        rows.append(["k = %s" % k,
                     f(d["diabetes"][k]["RAG"], 3), f(d["diabetes"][k]["GraphRAG"], 3),
                     f(d["gold"][k]["RAG"], 3), f(d["gold"][k]["GraphRAG"], 3)])
    return ("Đồ thị giúp được ở miền y văn nhưng không giúp ở miền vàng (context_recall)",
            ["Số tài liệu lấy về", "Tiểu đường: RAG", "Tiểu đường: GraphRAG",
             "Vàng: RAG", "Vàng: GraphRAG"], rows)


def t_c5_neo4j():
    """Kết quả các truy vấn Cypher chạy thật trên Neo4j (load_neo4j.py)."""
    q = {i: t for i, t in enumerate(C5("neo4j_queries.json")["truy_van"])}

    r1 = q[0]["ket_qua"]
    loai = sorted({r["qua_loai_nut"] for r in r1})
    nut = sorted({r["qua_nut"] for r in r1})

    r2 = q[1]["ket_qua"]
    hs = [r["he_so"] for r in r2]
    am = sum(1 for h in hs if h < 0)

    r3 = q[2]["ket_qua"]
    thang6 = ", ".join("%s (%d%s)" % (r["nguyen_nhan"], r["so_tin"],
                                      "" if r["tuong_quan_cung_quy"] is None
                                      else "; corr %s" % f(r["tuong_quan_cung_quy"], 3))
                       for r in r3)

    r4 = {bool(r["co_chuoi_gia"]): r["so_tin"] for r in q[3]["ket_qua"]}
    tong = r4[True] + r4[False]

    return ("Truy vấn Cypher chạy trên Neo4j và kết quả",
            ["Câu hỏi", "Kết quả"],
            [["Hai mã cổ phiếu nào nối nhau qua một nút trung gian?",
              "%d cặp, tất cả đi qua nút %s `%s`; không cặp nào chung ngành hay nhóm ngành"
              % (len(r1), "/".join(loai), "/".join(nut))],
             ["Cạnh vàng – S&P 500 qua từng quý",
              "%d cạnh, hệ số từ %s đến %s; âm ở %d quý, dương ở %d quý"
              % (len(hs), f(min(hs), 3), f(max(hs), 3), am, len(hs) - am)],
             ["Tháng 6/2026 báo chí quy giá vàng cho ai, có cạnh tương quan cùng quý không?",
              thang6 + "; chỉ nút có chuỗi giá mới có cạnh để đối chiếu"],
             ["Năm 2026, bao nhiêu tin viện tới nguyên nhân không có chuỗi giá?",
              "%s / %s tin (%s%%)" % (n(r4[False]), n(tong), f(100.0 * r4[False] / tong, 1))]])
