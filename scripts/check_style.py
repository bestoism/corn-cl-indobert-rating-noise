"""
check_style.py: pemeriksaan gaya dan sintaks (Subbab aturan gaya).
  1. Mencari karakter em dash (U+2014) dan en dash (U+2013) di seluruh berkas teks proyek.
  2. Memeriksa sintaks seluruh berkas Python dan sel kode notebook.
  3. Mencari rujukan nomor subbab lama (III.E, III.I.3, 3.9.4, dan sejenisnya).
Menjalankan: python scripts/check_style.py <folder-proyek>
"""

import ast
import json
import os
import re
import sys

EM, EN = "\u2014", "\u2013"
OLD_REFS = [r"\bIII\.[A-Z]\b", r"\bIII\.[A-Z]\.\d+\b", r"\b3\.9\.[1-4]\b", r"\bSubbab 1\.6\b", r"\bBab 1\.6\b"]
TEXT_EXT = (".py", ".md", ".txt", ".ipynb", ".json", ".csv", ".gitignore")


def iter_files(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", "__pycache__", ".ipynb_checkpoints")]
        for f in fn:
            if f.endswith(TEXT_EXT) or f == ".gitignore":
                yield os.path.join(dp, f)


def main(root):
    n_em = n_en = 0
    problems = []
    for path in iter_files(root):
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        e1, e2 = s.count(EM), s.count(EN)
        n_em += e1
        n_en += e2
        if e1 or e2:
            problems.append(f"{path}: em dash={e1}, en dash={e2}")
        if path.endswith(".py") and "check_style.py" not in path:
            try:
                ast.parse(s)
            except SyntaxError as ex:
                problems.append(f"{path}: kesalahan sintaks {ex}")
            for pat in OLD_REFS:
                if re.search(pat, s):
                    problems.append(f"{path}: rujukan subbab lama ditemukan ({pat})")
        if path.endswith(".ipynb"):
            nb = json.loads(s)
            for i, c in enumerate(nb["cells"]):
                src = "".join(c["source"])
                if c["cell_type"] == "code":
                    clean = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("!", "%")))
                    try:
                        ast.parse(clean)
                    except SyntaxError as ex:
                        problems.append(f"{path}: sel {i} kesalahan sintaks {ex}")
                for pat in OLD_REFS:
                    if re.search(pat, src):
                        problems.append(f"{path}: sel {i} rujukan subbab lama ({pat})")
    print(f"Karakter em dash : {n_em}")
    print(f"Karakter en dash : {n_en}")
    if problems:
        print("[BERHENTI] Masalah ditemukan:")
        for p in problems:
            print("  -", p)
        return 1
    print("[OK] Tidak ada em dash, en dash, kesalahan sintaks, atau rujukan subbab lama.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
