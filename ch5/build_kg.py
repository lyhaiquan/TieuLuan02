# -*- coding: utf-8 -*-
"""Dựng đồ thị tri thức thành đối tượng đồ thị thật, rồi xuất ra Neo4j.

Các script trước ghi cạnh ra CSV. Ở đó đồ thị mới chỉ là một bảng quan hệ:
không duyệt được, không hỏi được câu nào cần đi qua nhiều bước. Script này
nạp các cạnh đó vào một `networkx.MultiDiGraph`, chạy vài truy vấn duyệt để
kiểm chứng đúng những khẳng định mà Chương 5 nêu, rồi xuất:

  results/kg.cypher          script nạp vào Neo4j (CREATE CONSTRAINT + MERGE)
  results/kg.graphml         định dạng chuẩn, mở được bằng Gephi, yEd
  results/kg_graph.json      số liệu cấu trúc và kết quả các truy vấn duyệt

Chạy:  python build_kg.py
"""
import json
import os

import networkx as nx
import pandas as pd

import fin_core as F

OUT = os.path.join(F.HERE, "results")

# Nhãn nút của miền chứng khoán; miền vàng lấy nhãn từ ENTITIES["kind"].
STOCK_LABEL = {"thuộc ngành": "Industry", "thuộc nhóm": "Sector", "ở nước": "Country"}


def build():
    """Gom bốn tệp cạnh thành một đồ thị có hướng, đa cạnh."""
    g = nx.MultiDiGraph()

    for node, meta in F.ENTITIES.items():
        g.add_node(node, label=meta["kind"], has_series=meta["series"] is not None,
                   domain="gold")
    for t in F.TICKERS:
        g.add_node(t, label="Stock", has_series=True, domain="stock")

    corr = pd.read_csv(os.path.join(OUT, "kg_corr_edges.csv"))
    for _, e in corr.iterrows():
        g.add_edge(e.src, e.dst, key="corr|%s|%s|%s" % (e.src, e.dst, e.period),
                   rel="correlates_with", period=str(e.period),
                   corr=float(e["corr"]), n_days=int(e.n_days), domain="gold")

    scorr = pd.read_csv(os.path.join(OUT, "stocks_corr_edges.csv"))
    for _, e in scorr.iterrows():
        g.add_edge(e.src, e.dst, key="scorr|%s|%s|%s" % (e.src, e.dst, e.period),
                   rel="correlates_with", period=str(e.period),
                   corr=float(e["corr"]), n_days=int(e.n_days), domain="stock")

    clm = pd.read_csv(os.path.join(OUT, "kg_claim_edges.csv"))
    for _, e in clm.iterrows():
        g.add_edge(e.src, e.dst, key="claim|%s|%s|%s" % (e.src, e.dst, e.period),
                   rel="claims_cause", period=str(e.period),
                   n_claims=int(e.n_claims), nhat_quan=int(e.nhat_quan),
                   mau_thuan=int(e.mau_thuan),
                   khong_kiem_chung=int(e.khong_kiem_chung), domain="gold")

    bt = pd.read_csv(os.path.join(OUT, "stocks_belongs_to.csv"))
    for _, e in bt.iterrows():
        g.add_node(e.dst, label=STOCK_LABEL[e.chi_tiet], has_series=False,
                   domain="stock")
        g.add_edge(e.src, e.dst, key="bt|%s|%s" % (e.src, e.dst),
                   rel="belongs_to", chi_tiet=e.chi_tiet, domain="stock")
    return g


def simple_view(g, rel, domain=None, period=None):
    """Đồ thị vô hướng đơn chỉ gồm một loại cạnh, để duyệt đường đi.

    Phải bỏ trùng cạnh: cặp GOLD-USD có 27 cạnh tương quan (mỗi quý một cạnh),
    nhưng khi hỏi "đi từ nút nào tới nút nào được" thì chúng là một bước.
    """
    h = nx.Graph()
    h.add_nodes_from(n for n, d in g.nodes(data=True)
                     if domain is None or d.get("domain") == domain)
    for u, v, d in g.edges(data=True):
        if d["rel"] != rel:
            continue
        if domain is not None and d.get("domain") != domain:
            continue
        if period is not None and not str(d.get("period", "")).startswith(period):
            continue
        h.add_edge(u, v)
    return h


def queries(g):
    """Những câu hỏi chỉ trả lời được bằng cách duyệt đồ thị."""
    out = {}

    # 1. Đường đi gián tiếp giữa hai mã cổ phiếu: không có cạnh trực tiếp,
    #    nhưng nối nhau qua nút ngành hoặc quốc gia.
    bt = simple_view(g, "belongs_to", domain="stock")
    paths = []
    ma = sorted(F.TICKERS)
    for i, a in enumerate(ma):
        for b in ma[i + 1:]:
            if bt.has_edge(a, b):
                continue
            try:
                p = nx.shortest_path(bt, a, b)
            except nx.NetworkXNoPath:
                continue
            paths.append({"tu": a, "den": b, "so_buoc": len(p) - 1,
                          "duong_di": " -> ".join(p)})
    out["duong_di_gian_tiep_co_phieu"] = {
        "so_cap": len(paths), "vi_du": paths[:3],
        "do_dai_nho_nhat": min([p["so_buoc"] for p in paths], default=None)}

    # 2. Vì sao đồ thị vàng không khai thác được multi-hop: mọi cặp nút có
    #    chuỗi giá đều đã nối trực tiếp, nên không còn đường đi nào để bắc cầu.
    cg = simple_view(g, "correlates_with", domain="gold")
    # Fed, lạm phát, địa chính trị không có chuỗi giá nên không bao giờ có
    # cạnh tương quan; tính cả chúng thì sẽ đếm nhầm thành "cặp cần bắc cầu".
    cg = cg.subgraph(n for n in cg if g.nodes[n]["has_series"]).copy()
    tong = cg.number_of_nodes() * (cg.number_of_nodes() - 1) // 2
    truc_tiep = cg.number_of_edges()
    out["do_thi_vang"] = {
        "so_nut_co_chuoi_gia": cg.number_of_nodes(),
        "so_cap_nut": tong,
        "so_cap_noi_truc_tiep": truc_tiep,
        "so_cap_phai_bac_cau": tong - truc_tiep,
        "bac_trung_binh": round(2 * truc_tiep / cg.number_of_nodes(), 2)}

    # 3. Nút trung tâm của đồ thị khẳng định: ai được báo chí viện tới nhiều nhất.
    cl = [(u, v, d) for u, v, d in g.edges(data=True) if d["rel"] == "claims_cause"]
    bac = {}
    for u, _, d in cl:
        bac[u] = bac.get(u, 0) + d["n_claims"]
    out["nguyen_nhan_duoc_vien_nhieu_nhat"] = [
        {"nut": k, "so_tin": v, "co_chuoi_gia": F.ENTITIES.get(k, {}).get("series") is not None}
        for k, v in sorted(bac.items(), key=lambda x: -x[1])]

    # 4. Cùng một cặp nút, cạnh tương quan đổi dấu qua các quý: đây là thứ một
    #    cạnh duy nhất không biểu diễn được.
    doi_dau = []
    for u, v, d in g.edges(data=True):
        if d["rel"] != "correlates_with" or d["domain"] != "gold":
            continue
        doi_dau.append((u, v, d["period"], d["corr"]))
    df = pd.DataFrame(doi_dau, columns=["src", "dst", "period", "corr"])
    cap = (df.groupby(["src", "dst"])["corr"]
             .agg(so_lat="count", am=lambda s: int((s < 0).sum()),
                  duong=lambda s: int((s > 0).sum())).reset_index())
    cap = cap[(cap.am > 0) & (cap.duong > 0)]
    out["cap_co_canh_ca_am_lan_duong"] = {
        "so_cap": int(len(cap)),
        "tong_so_cap": int(df.groupby(["src", "dst"]).ngroups),
        "vi_du": cap.sort_values("so_lat", ascending=False).head(3).to_dict("records")}

    # 5. Thống kê chung.
    rels = {}
    for _, _, d in g.edges(data=True):
        rels[d["rel"]] = rels.get(d["rel"], 0) + 1
    nhan = {}
    for _, d in g.nodes(data=True):
        nhan[d["label"]] = nhan.get(d["label"], 0) + 1
    out["tong_quan"] = {"so_nut": g.number_of_nodes(), "so_canh": g.number_of_edges(),
                        "theo_loai_canh": rels, "theo_nhan_nut": nhan,
                        "so_thanh_phan_lien_thong":
                            nx.number_connected_components(g.to_undirected())}
    return out


def esc(s):
    return str(s).replace("\\", "\\\\").replace("'", "\\'")


def to_cypher(g, path):
    """Xuất script nạp vào Neo4j. Dùng MERGE nên chạy lại nhiều lần vẫn đúng."""
    lines = ["// Đồ thị tri thức Chương 5 - sinh tự động bởi build_kg.py",
             "// Nạp:  cat kg.cypher | cypher-shell -u neo4j -p <mat khau>",
             "CREATE CONSTRAINT entity_id IF NOT EXISTS",
             "FOR (n:Entity) REQUIRE n.id IS UNIQUE;", ""]
    for n, d in g.nodes(data=True):
        # name trùng id: Neo4j Browser lấy thuộc tính name làm nhãn hiển thị.
        lines.append("MERGE (n:Entity:%s {id: '%s'}) SET n.name = '%s', "
                     "n.has_series = %s, n.domain = '%s';"
                     % (d["label"], esc(n), esc(n), str(bool(d["has_series"])).lower(),
                        esc(d["domain"])))
    lines.append("")
    for u, v, d in g.edges(data=True):
        props = ", ".join(
            "%s: %s" % (k, ("'%s'" % esc(val)) if isinstance(val, str) else val)
            for k, val in sorted(d.items()) if k != "rel")
        lines.append("MATCH (a:Entity {id: '%s'}), (b:Entity {id: '%s'}) "
                     "MERGE (a)-[r:%s {%s}]->(b);" % (esc(u), esc(v), d["rel"], props))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    g = build()
    q = queries(g)
    to_cypher(g, os.path.join(OUT, "kg.cypher"))

    gm = nx.DiGraph()                      # GraphML không nhận đa cạnh có khoá chuỗi
    gm.add_nodes_from(g.nodes(data=True))
    for u, v, d in g.edges(data=True):
        if not gm.has_edge(u, v):
            gm.add_edge(u, v, **{k: str(val) for k, val in d.items()})
    nx.write_graphml(gm, os.path.join(OUT, "kg.graphml"))

    with open(os.path.join(OUT, "kg_graph.json"), "w", encoding="utf-8") as f:
        json.dump(q, f, ensure_ascii=False, indent=2)
    print(json.dumps(q, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
