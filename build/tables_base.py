# -*- coding: utf-8 -*-
"""Bang va doan van phu thuoc ket qua: doc truc tiep results/*.json, khong nhap tay so lieu."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def J(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def f(x, nd=4):
    """So thap phan kieu Viet Nam."""
    return ("%.*f" % (nd, x)).replace(".", ",")


def pm(ms, nd=4, scale=1.0):
    return "%s ± %s" % (f(ms[0] * scale, nd), f(ms[1] * scale, nd))


def pc(x, nd=1):
    return f(100 * x, nd) + "%"


def n(x):
    """So nguyen co dau cham ngan cach hang nghin."""
    return "{:,}".format(int(round(x))).replace(",", ".")


def loc(rel, names):
    """So dong code (bo dong trong, dong chu thich va docstring) cua cac def/class theo ten."""
    src = (ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    total = 0
    for name in names:
        node = next(x for x in ast.walk(tree) if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name == name)
        skip = set()
        for sub in ast.walk(node):
            if isinstance(sub, (ast.ClassDef, ast.FunctionDef)) and sub.body and isinstance(sub.body[0], ast.Expr)                     and isinstance(getattr(sub.body[0], "value", None), ast.Constant) and isinstance(sub.body[0].value.value, str):
                skip.update(range(sub.body[0].lineno, sub.body[0].end_lineno + 1))
        for i in range(node.lineno, node.end_lineno + 1):
            l = lines[i - 1].strip()
            if l and not l.startswith("#") and i not in skip:
                total += 1
    return total
