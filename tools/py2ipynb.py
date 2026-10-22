"""Chuyen file .py dinh dang 'percent' (# %% / # %% [markdown]) thanh .ipynb."""
import json
import sys
from pathlib import Path


def convert(src: Path, dst: Path, kernel_name: str, kernel_display: str):
    lines = src.read_text(encoding="utf-8").splitlines()
    cells = []
    cur_type, cur = None, []

    def flush():
        if cur_type is None:
            return
        text = "\n".join(cur).strip("\n")
        if not text.strip():
            return
        if cur_type == "markdown":
            body = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l)
                    for l in text.splitlines()]
            cells.append(dict(cell_type="markdown", metadata={},
                              source="\n".join(body)))
        else:
            cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                              outputs=[], source=text))

    for line in lines:
        if line.startswith("# %%"):
            flush()
            cur_type = "markdown" if "[markdown]" in line else "code"
            cur = []
        else:
            cur.append(line)
    flush()

    nb = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name=kernel_display, language="python",
                        name=kernel_name),
        language_info=dict(name="python")),
        nbformat=4, nbformat_minor=5)
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{dst.name}: {len(cells)} cells")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    kernel = sys.argv[3] if len(sys.argv) > 3 else "python3"
    disp = sys.argv[4] if len(sys.argv) > 4 else "Python 3"
    convert(src, dst, kernel, disp)
