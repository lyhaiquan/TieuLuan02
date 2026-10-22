# -*- coding: utf-8 -*-
"""Dung tieu luan: chapters/*.md (co the chua {{...}}) -> tieuluan.md -> .docx (pandoc) -> .pdf (Word).

Cu phap trong chapters/*.md:
  {{FIG file.png | chu thich}}           hinh tu figs/
  {{TAB ten_ham}}                        bang do ham trong tables.py sinh ra (tra ve (caption, header, rows))
  {{CODE duong_dan | ten1,ten2}}         trich def/class bang AST tu file nguon (goc TieuLuan01)
  {{CODEL duong_dan | moc_dau | moc_cuoi}}  trich theo dong moc
  {{V bieu_thuc}}                        gia tri Python (dung ham trong tables.py), vd {{V f(r['x'],3)}}
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import tables as T  # noqa: E402

OUT_MD = HERE / "tieuluan.md"
import os
NAME = os.environ.get("TL_NAME", "tieuluan02_05_02_quanlh")
OUT_DOCX = HERE / (NAME + ".docx")
OUT_PDF = HERE / (NAME + ".pdf")

fig_no = [0]
tab_no = [0]
code_no = [0]


def read_src(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def obj(rel, names):
    src = read_src(rel)
    tree = ast.parse(src)
    lines = src.splitlines()
    out = []
    for name in names:
        node = next((n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), None)
        if node is None:
            raise KeyError("%s khong co trong %s" % (name, rel))
        start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
        out.append("\n".join(lines[start:node.end_lineno]))
    return "\n\n".join(out)


def lines_between(rel, first, last):
    src = read_src(rel).splitlines()
    i = next(i for i, l in enumerate(src) if first in l)
    j = next(j for j in range(i, len(src)) if last in src[j])
    return "\n".join(src[i:j + 1])


def md_table(header, rows, align=None):
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sub(m):
    kind, body = m.group(1), m.group(2).strip()
    if kind == "FIG":
        f, cap = [s.strip() for s in body.split("|", 1)]
        fig_no[0] += 1
        return "![Hình %d. %s](figs/%s){width=100%%}\n" % (fig_no[0], cap, f)
    if kind == "FIGW":
        f, w, cap = [s.strip() for s in body.split("|", 2)]
        fig_no[0] += 1
        return "![Hình %d. %s](figs/%s){width=%s%%}\n" % (fig_no[0], cap, f, w)
    if kind == "TAB":
        cap, header, rows = getattr(T, body)()
        tab_no[0] += 1
        return "**Bảng %d.** %s\n\n%s\n" % (tab_no[0], cap, md_table(header, rows, ["---"] + ["---:"] * (len(header) - 1)))
    if kind == "TABL":      # bang can le trai moi cot (chu)
        cap, header, rows = getattr(T, body)()
        tab_no[0] += 1
        return "**Bảng %d.** %s\n\n%s\n" % (tab_no[0], cap, md_table(header, rows))
    if kind == "CODE":
        rel, names = [s.strip() for s in body.split("|", 1)]
        return "```python\n%s\n```\n" % obj(rel, [n.strip() for n in names.split(",")])
    if kind == "CODEL":
        rel, a, b = [s.strip() for s in body.split("|")]
        return "```python\n%s\n```\n" % lines_between(rel, a, b)
    if kind == "V":
        return str(eval(body, vars(T)))
    raise ValueError(kind)


def mark_new(txt):
    """Boi do phan noi dung moi them vao o Tieu luan 02.

    Chuong nao mo dau bang <!-- MOI --> thi moi doan van xuoi trong do duoc boc
    bang fenced div mang kieu MoiPara (chu do dinh nghia trong reference docx).
    Chi boc doan van: tieu de, bang, khoi code, hinh va cac macro {{...}} giu
    nguyen dinh dang, neu khong Word se mat kieu tieu de va kieu code.
    """
    if "<!-- MOI -->" not in txt:
        return txt
    txt = txt.replace("<!-- MOI -->\n", "")
    out, fences = [], 0
    for block in txt.split("\n\n"):
        s = block.strip()
        # Dem so dong mo/dong hang rao ``` da gap: le nghia la dang o trong khoi
        # code. Dem nhu vay moi dung khi khoi code co dong trong ben trong, vi
        # luc do no bi tach thanh nhieu "doan".
        in_code = fences % 2 == 1
        fences += len(re.findall(r"(?m)^```", block))
        if (not s or in_code or s.startswith("#") or s.startswith("```")
                or s.startswith("|") or s.startswith("{{") or s.startswith(":::")
                or s.startswith("!") or s.startswith(">")
                or s.startswith("$$")):
            out.append(block)
        else:
            out.append(':::: {custom-style="MoiPara"}\n%s\n::::' % block)
    return "\n\n".join(out)


def main():
    parts = []
    only = [a.split("=")[1] for a in sys.argv if a.startswith("--only=")]
    for f in sorted((HERE / "chapters").glob("*.md")):
        if only and not any(f.name.startswith(p) for p in only[0].split(",")):
            continue
        txt = mark_new(f.read_text(encoding="utf-8"))
        parts.append(re.sub(r"\{\{(FIGW|FIG|TABL|TAB|CODEL|CODE|V) (.*?)\}\}", sub, txt, flags=re.S))
    brk = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n'
    parts = [re.sub(r"(?m)^(# )", lambda m: brk + m.group(1), p, count=1) for p in parts]   # moi chuong o trang moi
    md = "\n\n".join(parts)
    left = re.findall(r"\{\{.*?\}\}", md)
    if left:
        raise SystemExit("cho trong chua dien: %s" % left[:5])
    # Dong tac gia (co ma sinh vien) nam ngoai repo cong khai.
    af = HERE / "author.local.txt"
    md = md.replace("AUTHOR_LINE", af.read_text(encoding="utf-8").strip()
                    if af.exists() else "Ly Hai Quan")
    OUT_MD.write_text(md, encoding="utf-8")
    print("md: %d ky tu, %d hinh, %d bang" % (len(md), fig_no[0], tab_no[0]))
    ref = HERE / "reference_code.docx"
    cmd = ["pandoc", str(OUT_MD), "-o", str(OUT_DOCX), "--toc", "--toc-depth=2", "--syntax-highlighting=tango",
           "--resource-path", str(ROOT), "--reference-doc", str(ref)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print("docx: %.0f KB" % (OUT_DOCX.stat().st_size / 1024))
    if "--no-pdf" in sys.argv:
        return
    ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "export_pdf.ps1"),
          "-DocPath", str(OUT_DOCX), "-PdfPath", str(OUT_PDF)]
    r = subprocess.run(ps, capture_output=True, text=True, timeout=900)
    print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
