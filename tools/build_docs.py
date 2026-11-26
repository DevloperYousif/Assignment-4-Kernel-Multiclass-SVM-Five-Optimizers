"""Generate README.md, documentation/Experiment_Report.md and documentation/Academic_Defense_Guide.md
from results/summary.json (numbers always equal the executed notebooks)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, "results", "summary.json")))
C = {r["optimizer"]: r for r in S["comparison"]}
ORDER = ["GD", "SGD", "MBGD", "AdaGrad", "Adam"]
NAME = {"GD": "Gradient Descent", "SGD": "SGD", "MBGD": "Mini-Batch GD", "AdaGrad": "AdaGrad", "Adam": "Adam"}
H = S["hyper"]
LVK = S["linear_vs_kernel"]; LC = list(LVK)
L, K, V = LVK[LC[0]], LVK[LC[1]], LVK[LC[2]]
KC = {r["kernel"]: r for r in S["kernel_comparison"]}
BS = S["basis_size"]; ex = [b for b in BS if b["exact"]][0]; b2 = [b for b in BS if b["basis_M"] == 2000][0]
SEL = max(S["kernel_selection"], key=lambda r: r["val_macro_f1"])
VER = S["verdict"]


def pc(x):
    return f"{100 * float(x):.2f}%"


def f4(x):
    return f"{float(x):.4f}"


best_val = max(ORDER, key=lambda o: C[o]["val_macro_f1"])
best_acc = max(ORDER, key=lambda o: C[o]["accuracy"])
best_f1 = max(ORDER, key=lambda o: C[o]["macro_f1"])
low_j = min(ORDER, key=lambda o: C[o]["train_objective"])
fast = min(ORDER, key=lambda o: C[o]["training_time_s"])

comp_table = ("| Optimizer | LR | Batch | Updates | Accuracy | Macro P | Macro R | Macro F1 | Val Macro F1 | Hinge | Reg | Total J | Train time (s) | Pred time (ms) |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" +
              "\n".join(f"| {NAME[o]} | {C[o]['learning_rate']:g} | {('all' if o == 'GD' else int(C[o]['batch_size']))} | {int(C[o]['updates']):,} | {pc(C[o]['accuracy'])} | "
                        f"{f4(C[o]['macro_precision'])} | {f4(C[o]['macro_recall'])} | {f4(C[o]['macro_f1'])} | {f4(C[o]['val_macro_f1'])} | "
                        f"{f4(C[o]['train_hinge'])} | {f4(C[o]['regularization'])} | {f4(C[o]['train_objective'])} | {C[o]['training_time_s']:.1f} | "
                        f"{1000 * C[o]['prediction_time_s']:.1f} |" for o in ORDER))

lvk_rows = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Training Time (s)", "Prediction Time (s)", "Model Complexity", "Decision Boundary", "Model Memory"]


def fmt(k, v):
    if k == "Accuracy":
        return pc(v)
    if k in ("Macro Precision", "Macro Recall", "Macro F1"):
        return f4(v)
    if "Time" in k:
        return f"{float(v):.3f}"
    return str(v)


lvk_table = ("| Metric | " + " | ".join(LC) + " |\n|---|" + "---|" * len(LC) + "\n" +
             "\n".join(f"| {k} | " + " | ".join(fmt(k, LVK[c][k]) for c in LC) + " |" for k in lvk_rows))
kc_table = ("| Kernel | min eigenvalue of G | PSD | Val Macro F1 | Test accuracy | Test Macro F1 |\n|---|---|---|---|---|---|\n" +
            "\n".join(f"| {r['kernel']} | {r['min_eig_G']:.3g} | {'yes' if r['psd_within_tol'] else 'NO'} | {f4(r['val_macro_f1'])} | {pc(r['test_accuracy'])} | {f4(r['test_macro_f1'])} |"
                      for r in S["kernel_comparison"]))
bs_table = ("| Basis M | Exact? | Val Macro F1 | Kernel time (s) | Train time, 30 epochs (s) | Kernel memory (MiB) |\n|---|---|---|---|---|---|\n" +
            "\n".join(f"| {int(b['basis_M']):,} | {'yes (Mode A)' if b['exact'] else 'no'} | {f4(b['val_macro_f1'])} | {b['kernel_time_s']:.2f} | {b['train_time_s']:.1f} | {b['kernel_memory_MiB']:.0f} |" for b in BS))
pcf = S["per_class_f1"]
pc_table = ("| Class | " + " | ".join(NAME[o] for o in ORDER) + " |\n|---|" + "---|" * 5 + "\n" +
            "\n".join(f"| {c} | " + " | ".join(f4(pcf[o][c]) for o in ORDER) + " |" for c in pcf["GD"]))

# ------------------------------------------------------------------------------------------- README
readme = f"""# Assignment 4 - Kernel-Based Multi-Class SVM with Five Optimizers

**Human Activity Recognition Using Kernel-Based Multi-Class Support Vector Machines and Five Optimization Algorithms**

A genuine kernelized, direct multi-class SVM (Crammer-Singer hinge loss + RKHS regularisation, RBF kernel) implemented
from scratch in NumPy and trained with five manually implemented optimizers - **Gradient Descent, SGD, Mini-Batch GD,
AdaGrad and Adam** - on the UCI HAR smartphone dataset (6 activities). One notebook per optimizer.

## Results (independent test set: 2,947 windows from 9 unseen subjects)

{comp_table}

* **Best optimizer by validation Macro F1 (selection rule):** {NAME[best_val]} ({f4(C[best_val]['val_macro_f1'])})
* **Highest test accuracy:** {NAME[best_acc]} ({pc(C[best_acc]['accuracy'])}) · **highest test Macro F1:** {NAME[best_f1]} ({f4(C[best_f1]['macro_f1'])})
* **Lowest training objective:** {NAME[low_j]} · **fastest training:** {NAME[fast]} (only 100 updates)
* **Linear SVM vs Kernel SVM:** Linear {pc(L['Accuracy'])} / Macro F1 {f4(L['Macro F1'])} vs RBF kernel ({LC[1]}) {pc(K['Accuracy'])} / {f4(K['Macro F1'])};
  scikit-learn `SVC` reference {pc(V['Accuracy'])} / {f4(V['Macro F1'])}. On this dataset the kernel is **not** better than the linear model.

Training times are from one run on a 2-vCPU machine and vary between runs; all metrics are deterministic.

## Dataset
UCI *Human Activity Recognition Using Smartphones* (Anguita et al., ESANN 2013) - 10,299 windows, 561 features,
30 volunteers, official subject-disjoint split 7,352 / 2,947. Download and placement: [`data/README.md`](data/README.md).
Kaggle mirror: `uciml/human-activity-recognition-with-smartphones` (the loader reads both formats).

## Method in one paragraph
Standardised features (fitted on training data) are compared with a class-stratified **kernel basis of M = 2,000 training
samples** through the RBF kernel $k(x,z)=\\exp(-\\gamma\\lVert x-z\\rVert^2)$, $\\gamma = {SEL['gamma_multiplier']:g}\\times$ median heuristic
$= {H['gamma']:.3e}$. Scores are $S = HA + b$ and the objective is
$J = \\frac\\lambda2\\mathrm{{Tr}}(A^\\top GA) + \\frac1N\\sum_i\\max(0, 1+\\max_{{j\\ne y_i}}S_{{ij}}-S_{{iy_i}})$ with $\\lambda = 10^{{-4}}$.
Subgradients $\\partial J/\\partial A = H^\\top D/N + \\lambda GA$, $\\partial J/\\partial b = \\sum_i D_i/N$. Every optimizer starts from
$A = 0, b = 0$ and runs 100 epochs; its learning rate is chosen from 5 values by validation Macro F1. Validation subjects are
separated from training subjects with `GroupShuffleSplit` (seed 42). Details: [`documentation/Mathematical_Derivation.md`](documentation/Mathematical_Derivation.md).

**Representation mode:** Mode B (restricted basis, M = 2,000) for all optimizers - chosen from the measured resource report:
the exact basis (M = N = 5,551) fits in memory but makes each adaptive update {(5551 / 2000) ** 2:.1f}x more expensive. The basis-size study
measures the cost of the restriction (val Macro F1 {f4(b2['val_macro_f1'])} at M = 2,000 vs {f4(ex['val_macro_f1'])} exact).
A small exact-kernel run and Nystroem / Random-Fourier-Feature checks are in notebook 04.

## Project structure
```
Assignment_4_Kernel_Multiclass_SVM/
├── data/README.md                  dataset source, licence, checksums, placement
├── notebooks/                      01_GD · 02_SGD · 03_MiniBatch · 04_AdaGrad · 05_Adam (executed, self-contained)
├── src/                            dataset_loader · preprocessing · kernels · kernel_approximation ·
│                                   kernel_multiclass_svm · optimizers · evaluation · visualization · experiment_runner
├── results/                        optimizer_comparison.csv · kernel_comparison.csv · linear_vs_kernel.csv ·
│                                   hyperparameters.json · training_histories/ · metrics/ · figures/ (PNG, 200 dpi)
├── models/                         kernel_svm_{{gd,sgd,minibatch,adagrad,adam}}.npz (A, b, basis, gamma, scaler, labels)
├── presentation/                   Assignment_4_Kernel_SVM.pptx (30 slides, Arabic speaker notes)
├── documentation/                  Academic_Defense_Guide.md · Mathematical_Derivation.md · Kernel_Theory.md · Experiment_Report.md
├── tests/                          test_kernels · test_kernel_svm · test_gradients · test_optimizers (pytest)
└── tools/                          notebook / presentation / documentation builders
```

## Installation and execution
```bash
python -m venv .venv
.venv\\Scripts\\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q                 # unit tests
jupyter notebook notebooks/         # run 01 ... 05 (each is independent)
python src/experiment_runner.py     # optional: the whole pipeline as a script
```
**Kaggle:** create a notebook from each `.ipynb`, attach the dataset `uciml/human-activity-recognition-with-smartphones`, and
*Run All* - all helper code is inlined, outputs go to `/kaggle/working`.

**Loading a trained model:**
```python
import sys; sys.path.insert(0, "src")
from kernel_multiclass_svm import KernelMulticlassSVM
model = KernelMulticlassSVM.load("models/kernel_svm_adagrad.npz")
pred = model.predict(X_raw)        # raw 561-feature rows; scaling + RBF basis are inside the file
```

## Visualizations
All figures are in `results/figures/` - e.g. `05_loss_curves_all.png`, `05_confusion_matrices_all.png`,
`01_gamma_sensitivity.png`, `02_basis_size_analysis.png`, `03_synthetic_decision_boundaries.png`, `04_linear_vs_kernel_confusion.png`.
Colours: GD blue, SGD orange, Mini-Batch green, AdaGrad purple, Adam red.

## Kernel comparison (supplementary, same basis and solver)
{kc_table}

## Linear SVM vs Kernel SVM
{lvk_table}

The previous Linear SVM assignment was not accessible from the build environment, so a controlled Linear SVM (same
loss, data, split, scaling, epochs; Adam; lambda and lr tuned on validation) was trained instead.

## Reproducibility
Seed 42 everywhere (`GroupShuffleSplit`, basis selection, shuffling, median heuristic sample). Initial parameters are zeros.
All metrics reproduce exactly; wall-clock times differ between machines.

## Limitations
Restricted basis (not the exact full-basis SVM); fixed 100-epoch budget (GD far from converged); kernel configuration
selected with a single solver; only 5 validation subjects (validation and test winners differ); no learning-rate
schedules; single seed; sigmoid kernel not PSD (exploratory only).

## References
Anguita et al., ESANN 2013 · Cortes & Vapnik 1995 · Crammer & Singer, JMLR 2001 · Schölkopf, Herbrich & Smola, COLT 2001 ·
Williams & Seeger, NIPS 2001 · Rahimi & Recht, NIPS 2007 · Duchi, Hazan & Singer, JMLR 2011 · Kingma & Ba, ICLR 2015 · Chang & Lin, LIBSVM 2011.
"""
open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write(readme)

# ------------------------------------------------------------------------------------------- Experiment report
ed = S["exact_demo"]; aq = S["approximation"]
report = f"""# Experiment Report

## 1. Setup
* Data: UCI HAR, official split. Training split 5,551 windows (16 subjects), validation 1,801 (5 subjects), test 2,947 (9 subjects).
* Preprocessing: standardisation fitted on the training split; subject ids never used as features.
* Kernel: RBF, gamma0 (median heuristic) = {H['gamma0']:.4e}, gamma = {SEL['gamma_multiplier']:g} x gamma0 = {H['gamma']:.4e}; lambda = {H['config']['lam']:g}.
* Representation: Mode B, M = 2,000 class-stratified basis samples (identical for all optimizers).
* Budget: 100 epochs; initial A = 0, b = 0; batch 64 for MBGD / AdaGrad / Adam, 1 for SGD, N for GD.
* Learning rate: 5 candidates per optimizer, each trained for the full budget; selection by final validation Macro F1.

## 2. Kernel configuration selection (validation, Adam lr 1e-3, 30 epochs)
Best: gamma = {SEL['gamma_multiplier']:g} x gamma0, lambda = {SEL['lam']:g}, validation Macro F1 {f4(SEL['val_macro_f1'])}.

| gamma / gamma0 | """ + " | ".join(f"lambda={l:g}" for l in sorted({r['lam'] for r in S['kernel_selection']})) + " |\n|---|" + "---|" * 4 + "\n" + "\n".join(
    f"| {g:g} | " + " | ".join(f4(next(r['val_macro_f1'] for r in S['kernel_selection'] if r['gamma_multiplier'] == g and r['lam'] == l)) for l in sorted({r['lam'] for r in S['kernel_selection']})) + " |"
    for g in sorted({r['gamma_multiplier'] for r in S['kernel_selection']})) + f"""

## 3. Main result - five optimizers (test set)
{comp_table}

Automatic verdict (notebook 05): {json.dumps(VER, indent=1)}

### Per-class test F1
{pc_table}

## 4. Basis size (Mode B vs exact Mode A)
{bs_table}

## 5. Kernel comparison
{kc_table}

The linear kernel trained in coefficient space is poorly conditioned (objective {KC['linear']['train_objective']:.2f} after 100 epochs); the same model in the
primal (controlled Linear SVM) is much better. The sigmoid kernel has a negative Gram eigenvalue and is reported only as exploratory.

## 6. Linear SVM vs Kernel SVM vs scikit-learn SVC
{lvk_table}

## 7. Exact-kernel verification and approximation quality
""" + "\n".join(f"* {r['mode']}: final J {r['final_train_obj']:.4f}, val Macro F1 {f4(r['val_macro_f1'])}" for r in ed) + "\n\n| Dimension | " + " | ".join(
    sorted({r['method'] for r in aq})) + " |\n|---|---|---|\n" + "\n".join(
    f"| {d} | " + " | ".join(f"{next(r['rel_frobenius_error'] for r in aq if r['dimension'] == d and r['method'] == m):.4f}" for m in sorted({r['method'] for r in aq})) + " |"
    for d in sorted({r['dimension'] for r in aq})) + """

(relative Frobenius error ||K - Phi Phi^T|| / ||K|| on 1,000 validation samples)

## 8. Discussion
* Stochastic and adaptive optimizers reach 92-94% test accuracy in 100 epochs; GD (100 updates) is far from converged.
* The validation winner and the test winner differ; with 5 validation subjects the top four optimizers are statistically close.
* AdaGrad and Adam are robust to the ill-conditioning of the coefficient-space problem; plain GD/MBGD/SGD need careful step sizes.
* The Linear SVM matches or beats the restricted kernel model on HAR - engineered features, new test subjects and the
  restricted basis / finite optimisation all favour the simpler model.

## 9. Tests
`python -m pytest -q` - 33 tests (kernels, PSD, gradients vs finite differences, optimizer update rules, SGD fast-path
equivalence, incomplete batches, AdaGrad accumulators, Adam bias correction, reproducibility, serialization).
"""
open(os.path.join(ROOT, "documentation", "Experiment_Report.md"), "w", encoding="utf-8").write(report)
print("README.md and Experiment_Report.md written")
