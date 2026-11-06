# -*- coding: utf-8 -*-
"""Nạp đồ thị vào Neo4j rồi chạy các truy vấn Cypher minh hoạ cho Chương 5.

Cần một Neo4j đang chạy. Cách nhanh nhất là Docker (xem README):
  docker run -d --name kg-neo4j -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/<mat khau> neo4j:5

Thông tin kết nối đọc từ biến môi trường NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
hoặc từ tệp ch5/neo4j.local.json (đã gitignore, không đẩy lên repo).

Chạy:  python build_kg.py && python load_neo4j.py
"""
import json
import os

from neo4j import GraphDatabase

import fin_core as F

OUT = os.path.join(F.HERE, "results")

# Mỗi truy vấn kèm câu hỏi bằng lời để người đọc biết nó trả lời điều gì.
QUERIES = [
    ("Hai mã cổ phiếu nào nối nhau qua một nút trung gian (đường đi 2 bước)?",
     """
     MATCH (a:Stock)-[:belongs_to]->(x)<-[:belongs_to]-(b:Stock)
     WHERE a.id < b.id
     RETURN a.id AS ma_1, labels(x)[1] AS qua_loai_nut, x.id AS qua_nut, b.id AS ma_2
     ORDER BY qua_nut, ma_1, ma_2
     """),
    ("Cạnh vàng - S&P 500 thay đổi thế nào qua từng quý?",
     """
     MATCH (:Entity {id: 'GOLD'})-[r:correlates_with]-(:Entity {id: 'SP500'})
     RETURN r.period AS quy, r.corr AS he_so, r.n_days AS so_phien
     ORDER BY quy
     """),
    ("Tháng 6/2026 báo chí quy giá vàng cho ai, và nguyên nhân đó có cạnh tương quan "
     "cùng kỳ để đối chiếu không?",
     """
     MATCH (c:Entity)-[k:claims_cause {period: '2026-06'}]->(g:Entity {id: 'GOLD'})
     OPTIONAL MATCH (g)-[r:correlates_with {period: '2026-06-30'}]-(c)
     RETURN c.id AS nguyen_nhan, k.n_claims AS so_tin, c.has_series AS co_chuoi_gia,
            r.corr AS tuong_quan_cung_quy
     ORDER BY so_tin DESC
     """),
    ("Trên toàn bộ năm 2026, bao nhiêu tin viện tới nguyên nhân không có chuỗi giá?",
     """
     MATCH (c:Entity)-[k:claims_cause]->(:Entity {id: 'GOLD'})
     RETURN c.has_series AS co_chuoi_gia, sum(k.n_claims) AS so_tin,
            collect(DISTINCT c.id) AS cac_nut
     ORDER BY co_chuoi_gia
     """),
    ("Đếm nút và cạnh theo loại trong cơ sở dữ liệu",
     """
     MATCH ()-[r]->() RETURN type(r) AS loai_canh, count(*) AS so_canh
     ORDER BY so_canh DESC
     """),
]


def config():
    path = os.path.join(F.HERE, "neo4j.local.json")
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    return (os.environ.get("NEO4J_URI", cfg.get("uri", "bolt://localhost:7687")),
            os.environ.get("NEO4J_USER", cfg.get("user", "neo4j")),
            os.environ.get("NEO4J_PASSWORD", cfg.get("password")))


def statements(path):
    """Tách script Cypher thành từng câu lệnh, bỏ dòng chú thích."""
    with open(path, encoding="utf-8") as f:
        body = "\n".join(l for l in f.read().splitlines() if not l.startswith("//"))
    return [s.strip() for s in body.split(";\n") if s.strip()]


def main():
    uri, user, pw = config()
    if not pw:
        raise SystemExit("Chưa có mật khẩu: đặt NEO4J_PASSWORD hoặc tạo ch5/neo4j.local.json")
    drv = GraphDatabase.driver(uri, auth=(user, pw))
    drv.verify_connectivity()

    stmts = statements(os.path.join(OUT, "kg.cypher"))
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")          # nạp lại từ đầu cho sạch
        s.run(stmts[0])                             # ràng buộc id duy nhất
        with s.begin_transaction() as tx:
            for st in stmts[1:]:
                tx.run(st)
            tx.commit()
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print("Đã nạp %d nút, %d cạnh vào %s" % (nodes, rels, uri))

    res = {"so_nut": nodes, "so_canh": rels, "truy_van": []}
    with drv.session() as s:
        for question, cypher in QUERIES:
            rows = [r.data() for r in s.run(cypher)]
            res["truy_van"].append({"cau_hoi": question,
                                    "cypher": " ".join(cypher.split()),
                                    "ket_qua": rows})
            print("\n# " + question)
            for r in rows[:8]:
                print("  ", r)
    drv.close()

    with open(os.path.join(OUT, "neo4j_queries.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
