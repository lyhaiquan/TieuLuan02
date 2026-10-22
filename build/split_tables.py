from pathlib import Path
h = Path(__file__).parent
s = (h / "tables.py").read_text(encoding="utf-8")
i = s.index("# ======================================================================= Mo dau")
base, rest = s[:i], s[i:]
(h / "tables_base.py").write_text(base, encoding="utf-8")
(h / "tables_c1.py").write_text('# -*- coding: utf-8 -*-\n"""Mo dau va Chuong 1."""\nfrom tables_base import J, f, pm, pc, n, loc\n\n\n' + rest, encoding="utf-8")
(h / "tables.py").write_text('# -*- coding: utf-8 -*-\nfrom tables_base import *  # noqa\nfrom tables_c1 import *  # noqa\nfrom tables_c3 import *  # noqa\ntry:\n    from tables_c2 import *  # noqa\nexcept ImportError:\n    pass\ntry:\n    from tables_c4 import *  # noqa\nexcept ImportError:\n    pass\n', encoding="utf-8")
print("ok")
