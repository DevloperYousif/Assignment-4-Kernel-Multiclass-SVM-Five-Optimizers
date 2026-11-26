"""Re-generate the 'Results Interpretation' and 'Conclusion' Markdown cells of the EXECUTED notebooks from the final
result files (Markdown cells carry no outputs, so no re-execution is needed)."""
import os, sys
import nbformat as nbf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from notebook_discussion import discussion
from build_notebooks import NB_FILES

for opt, fn in NB_FILES.items():
    path = os.path.join(ROOT, "notebooks", fn)
    nb = nbf.read(path, as_version=4)
    d = discussion(opt)
    n = 0
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.startswith("## 22. Results Interpretation"):
            c.source = "## 22. Results Interpretation\n" + d["interpretation"]; n += 1
        if c.cell_type == "markdown" and c.source.startswith("## 23. Conclusion"):
            c.source = "## 23. Conclusion\n" + d["conclusion"]; n += 1
    nbf.write(nb, path)
    print(fn, "patched", n)
