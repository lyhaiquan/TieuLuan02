# -*- coding: utf-8 -*-
"""Bật một Neo4j 5 trong Docker, dùng mật khẩu trong ch5/neo4j.local.json.

Tệp cấu hình chưa có thì tự tạo với mật khẩu ngẫu nhiên. Mật khẩu chỉ nằm
trong tệp đó (đã gitignore), không in ra màn hình và không nằm trong repo.

Chạy:  python start_neo4j.py
Rồi mở http://localhost:7474 để xem đồ thị trong Neo4j Browser.
"""
import json
import os
import secrets
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "neo4j.local.json")
NAME = "kg-neo4j"


def main():
    if not os.path.exists(CFG):
        with open(CFG, "w", encoding="utf-8") as f:
            json.dump({"uri": "bolt://localhost:7687", "user": "neo4j",
                       "password": "kg-" + secrets.token_urlsafe(12)}, f, indent=1)
    with open(CFG, encoding="utf-8") as f:
        cfg = json.load(f)

    running = subprocess.run(["docker", "ps", "-a", "-q", "-f", "name=^%s$" % NAME],
                             capture_output=True, text=True).stdout.strip()
    if running:
        subprocess.run(["docker", "start", NAME], check=True, capture_output=True)
    else:
        subprocess.run(["docker", "run", "-d", "--name", NAME,
                        "-p", "7474:7474", "-p", "7687:7687",
                        "-e", "NEO4J_AUTH=%s/%s" % (cfg["user"], cfg["password"]),
                        "neo4j:5"], check=True, capture_output=True)

    # Neo4j cần khoảng 10-30 giây mới nhận kết nối bolt.
    from neo4j import GraphDatabase
    for _ in range(60):
        try:
            with GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"])) as d:
                d.verify_connectivity()
            print("Neo4j sẵn sàng: http://localhost:7474 (bolt %s)" % cfg["uri"])
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("Neo4j không lên sau 2 phút; xem: docker logs %s" % NAME)


if __name__ == "__main__":
    main()
