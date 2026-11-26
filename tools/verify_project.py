"""Final quality checks: executed notebooks without errors, exactly five notebooks / optimizers, consistent results,
loadable models, presentation numbers equal to the result files."""
import glob
import json
import os
import re
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)


nbs = sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb")))
check(len(nbs) == 5, f"exactly five notebooks ({len(nbs)})")
for p in nbs:
    nb = json.load(open(p))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errs = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    unexec = [c for c in code if c.get("execution_count") is None]
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    secs = [f"## {i}." for i in range(2, 25)]
    check(not errs and not unexec, f"{os.path.basename(p)}: {len(code)} code cells executed, {len(errs)} errors")
    check(all(s in md for s in secs) and "# 1." in md, f"{os.path.basename(p)}: all 24 required sections present")
    src = "\n".join("".join(c["source"]) for c in code)
    local = re.findall(r"^\s*(?:from|import)\s+(?:src|dataset_loader|preprocessing|kernels|kernel_approximation|kernel_multiclass_svm|optimizers|evaluation|visualization|experiment_runner)\b", src, re.M)
    check(not local, f"{os.path.basename(p)}: self-contained (no local module imports)")

R = os.path.join(ROOT, "results")
comp = pd.read_csv(os.path.join(R, "optimizer_comparison.csv"))
check(list(comp.optimizer) == ["GD", "SGD", "MBGD", "AdaGrad", "Adam"], "five optimizers in the comparison")
for o in comp.optimizer:
    m = json.load(open(os.path.join(R, "metrics", f"{o}_test_metrics.json")))
    row = comp[comp.optimizer == o].iloc[0]
    check(abs(m["accuracy"] - row.accuracy) < 1e-12 and abs(m["macro_f1"] - row.macro_f1) < 1e-12,
          f"{o}: notebook metrics json == comparison table")
    h = pd.read_csv(os.path.join(R, "training_histories", f"{o}.csv"))
    check(len(h) == 101 and abs(h.train_obj.iloc[-1] - row.train_objective) < 1e-9, f"{o}: history has 101 rows and matches final J")
    cm = np.array(m["confusion_matrix"])
    check(cm.shape == (6, 6) and cm.sum() == 2947 and (cm.sum(1) > 0).all(), f"{o}: all 6 classes present in test confusion matrix")

from kernel_multiclass_svm import KernelMulticlassSVM  # noqa: E402
from dataset_loader import load_har  # noqa: E402
from preprocessing import encode_labels  # noqa: E402
raw = load_har()
Xte, yte = raw["X_test"].to_numpy(float), encode_labels(raw["y_test"])
for f, o in [("kernel_svm_gd", "GD"), ("kernel_svm_sgd", "SGD"), ("kernel_svm_minibatch", "MBGD"), ("kernel_svm_adagrad", "AdaGrad"), ("kernel_svm_adam", "Adam")]:
    m = KernelMulticlassSVM.load(os.path.join(ROOT, "models", f + ".npz"))
    acc = float(np.mean(m.predict(Xte) == yte))
    check(abs(acc - comp[comp.optimizer == o].accuracy.iloc[0]) < 1e-12, f"models/{f}.npz loads and reproduces test accuracy {acc:.4f}")

pp = os.path.join(ROOT, "presentation", "Assignment_4_Kernel_SVM.pptx")
if os.path.exists(pp):
    z = zipfile.ZipFile(pp)
    slides = [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)]
    notes = [n for n in z.namelist() if re.match(r"ppt/notesSlides/notesSlide\d+\.xml$", n)]
    check(25 <= len(slides) <= 30 and len(notes) == len(slides), f"presentation: {len(slides)} slides, {len(notes)} notes")
    text = " ".join(re.sub("<[^>]+>", " ", z.read(n).decode("utf8")) for n in slides)
    for o in comp.optimizer:
        r = comp[comp.optimizer == o].iloc[0]
        check(f"{100 * r.accuracy:.2f}%" in text and f"{r.macro_f1:.4f}" in text, f"presentation shows {o} accuracy/F1 from the results")
    rtl = all('rtl="1"' in z.read(n).decode("utf8") for n in notes)
    check(rtl, "all speaker notes are right-to-left Arabic paragraphs")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
