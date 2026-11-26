"""Generate the five self-contained (Kaggle-ready) notebooks.

All supporting code from src/ is INLINED into the notebooks (Kaggle cannot import local modules).
Lines tagged '# local-import' are removed when inlining. Discussion text containing numbers is filled
from results/*.csv produced by `python src/experiment_runner.py` and re-checked after execution.
Usage:  python tools/build_notebooks.py
"""
import copy
import json
import os
import sys
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import notebook_text as T  # noqa: E402
from notebook_discussion import discussion  # noqa: E402

NB_FILES = {"GD": "01_Kernel_SVM_GD.ipynb", "SGD": "02_Kernel_SVM_SGD.ipynb", "MBGD": "03_Kernel_SVM_MiniBatch.ipynb",
            "AdaGrad": "04_Kernel_SVM_AdaGrad.ipynb", "Adam": "05_Kernel_SVM_Adam.ipynb"}
PREFIX = {"GD": "01", "SGD": "02", "MBGD": "03", "AdaGrad": "04", "Adam": "05"}


def module_code(name):
    src = open(os.path.join(ROOT, "src", f"{name}.py"), encoding="utf-8").read()
    if name == "experiment_runner":
        src = src.split("# ----------------------------------------------------------------------------- full pipeline (CLI)")[0]
    lines = [l for l in src.splitlines() if "# local-import" not in l]
    return f"# ===== inlined from src/{name}.py =====\n" + "\n".join(lines).strip() + "\n"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


SETUP = r'''
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
np.set_printoptions(precision=4, suppress=True)

# ---- output folders: /kaggle/working on Kaggle, the project root locally
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    OUT = "/kaggle/working"
else:
    OUT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
RESULTS = os.path.join(OUT, "results"); MODELS = os.path.join(OUT, "models")
for p in (RESULTS, MODELS, os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "metrics"),
          os.path.join(RESULTS, "training_histories")):
    os.makedirs(p, exist_ok=True)
print("NumPy", np.__version__, "| pandas", pd.__version__, "| matplotlib", matplotlib.__version__)
print("Kaggle:", ON_KAGGLE, "| outputs ->", OUT)
'''

EDA = r'''
raw = load_har()
print("layout:", raw["layout"], "| path:", raw["path"])
display(describe_partition(raw))
df_tr = raw["X_train"].copy(); df_tr["Activity"] = raw["y_train"].values; df_tr["subject"] = raw["subject_train"].values
print("Shape of the official training table (features + Activity + subject):", df_tr.shape)
show_cols = list(df_tr.columns[:6]) + ["subject", "Activity"]
print("First five records (first 6 of 561 features):"); display(df_tr[show_cols].head())
print("Last five records:"); display(df_tr[show_cols].tail())
'''

EDA2 = r'''
X_all = raw["X_train"]
print("Data types:", dict(X_all.dtypes.value_counts()))
print("Missing values (train / test):", int(X_all.isna().sum().sum()), "/", int(raw["X_test"].isna().sum().sum()))
print("Duplicated feature rows in train:", int(X_all.duplicated().sum()))
print("Feature value range: [%.3f, %.3f] (the authors already scaled features to [-1, 1])" % (X_all.values.min(), X_all.values.max()))
REP_FEATURES = ["tBodyAcc-mean()-X", "tGravityAcc-mean()-X", "tBodyAccMag-mean()", "fBodyAccJerk-std()-X",
                "tBodyGyro-std()-X", "angle(X,gravityMean)"]
display(X_all[REP_FEATURES].describe().T.round(4))
counts = raw["y_train"].value_counts().reindex(ACTIVITY_LABELS)
dist = pd.DataFrame({"train": counts, "train %": (100 * counts / counts.sum()).round(2),
                     "test": raw["y_test"].value_counts().reindex(ACTIVITY_LABELS)})
display(dist)
print("Imbalance ratio (largest / smallest class, train): %.2f" % (counts.max() / counts.min()))
'''

EDA3 = r'''
set_fig_dir(os.path.join(RESULTS, "figures"))
class_distribution(raw["y_train"], raw["y_test"], ACTIVITY_LABELS, name="eda_class_distribution.png")
feature_distributions(X_all, raw["y_train"], REP_FEATURES[:4], ACTIVITY_LABELS, name="eda_feature_distributions.png")
CORR_FEATURES = REP_FEATURES + ["tBodyAcc-std()-X", "tBodyAccJerk-mean()-X", "tBodyGyroMag-mean()",
                                "fBodyGyro-meanFreq()-X", "tGravityAcc-mean()-Y", "angle(Y,gravityMean)"]
correlation_heatmap(X_all, CORR_FEATURES, name="eda_feature_correlation.png")
# Outliers: share of standardised values with |z| > 3 per feature (computed on the official training set)
z = (X_all - X_all.mean()) / X_all.std().replace(0, 1)
out_share = (z.abs() > 3).mean().sort_values(ascending=False)
print("Overall share of |z| > 3 values: %.3f%%" % (100 * (z.abs() > 3).values.mean()))
display(out_share.head(8).rename("share |z|>3").to_frame().round(4))
print("Outliers are kept: they are real sensor readings, and the RBF kernel is bounded (0 < k <= 1),")
print("so a single extreme point cannot dominate the decision scores.")
'''

PREP = r'''
y_full = encode_labels(raw["y_train"]); y_test_enc = encode_labels(raw["y_test"])
subjects = raw["subject_train"].to_numpy()
tr_idx, va_idx = subject_group_split(subjects, val_fraction=0.2, seed=SEED)
print("training subjects  :", sorted(np.unique(subjects[tr_idx])))
print("validation subjects:", sorted(np.unique(subjects[va_idx])))
print("test subjects      :", sorted(raw["subject_test"].unique()))
assert set(subjects[tr_idx]).isdisjoint(subjects[va_idx]), "subject leakage!"
assert set(subjects).isdisjoint(set(raw["subject_test"])), "official partition broken!"
scaler = Standardizer().fit(raw["X_train"].to_numpy(float)[tr_idx])          # training split ONLY
X_tr_check = scaler.transform(raw["X_train"].to_numpy(float)[tr_idx])
print("sizes  train / val / test:", len(tr_idx), "/", len(va_idx), "/", len(y_test_enc))
props = pd.DataFrame({"train": np.bincount(y_full[tr_idx]) / len(tr_idx), "validation": np.bincount(y_full[va_idx]) / len(va_idx),
                      "test": np.bincount(y_test_enc) / len(y_test_enc)}, index=ACTIVITY_LABELS).round(3)
display(props)
print("standardised train: mean %.2e, std %.3f" % (X_tr_check.mean(), X_tr_check.std()))
gamma0_check = median_heuristic_gamma(X_tr_check, seed=SEED)
print("median-heuristic gamma0 = %.4e" % gamma0_check)
pca_ev = pca_projection(X_tr_check, y_full[tr_idx], ACTIVITY_LABELS, name="eda_pca_projection.png")
'''

KERNEL_DEMO = r'''
# Quick check of the kernels on three standardised training windows (one WALKING, one SITTING, one LAYING)
ex = np.r_[np.flatnonzero(y_full[tr_idx] == 0)[:1], np.flatnonzero(y_full[tr_idx] == 3)[:1], np.flatnonzero(y_full[tr_idx] == 5)[:1]]
Xe = X_tr_check[ex]
for kname, kf in [("linear", make_kernel("linear")), ("polynomial d=2", make_kernel("polynomial", degree=2)),
                  ("rbf (gamma0)", make_kernel("rbf", gamma=gamma0_check))]:
    print(kname); print(np.round(kf(Xe, Xe), 3))
print("RBF diagonal = 1 (every sample is maximally similar to itself); off-diagonal values shrink with distance.")
'''

CONFIG_CELL = r'''
print("Shared configuration (identical in all five notebooks):"); print(json.dumps(CONFIG, indent=2))
print("Learning-rate grids (5 values each = equal tuning budget):"); print(LR_GRID)
rep, info = resource_report(len(tr_idx), X_tr_check.shape[1], CONFIG["n_basis"], len(va_idx), len(y_test_enc))
display(rep); print(info)
'''

MODE_MD = r"""### Choice of the kernel representation (measured)
The table above shows that the **exact** representation (Mode A, $M = N$) would fit in memory, but every
mini-batch update of AdaGrad / Adam / MBGD needs $\lambda G A$, which costs $2M^2K$ FLOPs - about
**{exact_ratio:.1f}x** more for $M = N$ than for $M = 2000$ - and this is repeated for 5 optimizers x 5 learning
rates x 100 epochs. We therefore use **Mode B - a reduced kernel basis of M = 2000 class-stratified training
samples**, identical for all five optimizers. This is a *restricted* kernel expansion (the Representer Theorem
solution projected on 2,000 basis functions), **not** the exact full-basis SVM. Notebook 02 measures the cost of this
restriction (basis size up to the exact $M = N$), and notebook 04 runs a small exact-kernel verification and a
Nystroem / Random-Fourier-Feature approximation check."""

PREPARE = r'''
d = prepare_data()            # same steps as Section 9 + cached kernel matrices H_tr, H_va, H_te and G
assert np.allclose(d["X_tr"], X_tr_check) and np.array_equal(d["tr_idx"], tr_idx)
print("gamma = %.1f x %.4e = %.4e | lambda = %g | kernel evaluation time %.2f s"
      % (CONFIG["gamma_multiplier"], d["gamma0"], d["gamma"], CONFIG["lam"], d["kernel_time"]))
print("H_tr", d["H_tr"].shape, "H_va", d["H_va"].shape, "H_te", d["H_te"].shape, "G", d["G"].shape)
print("basis class counts:", dict(zip(ACTIVITY_LABELS, np.bincount(d["y_tr"][d["basis_idx"]]))))
print("G symmetric:", np.allclose(d["G"], d["G"].T), "| diag(G) = 1:", np.allclose(np.diag(d["G"]), 1),
      "| min eigenvalue of G: %.2e (PSD up to round-off)" % np.linalg.eigvalsh(d["G"])[0])
'''


def train_cells(opt, pre, selected_lr):
    return [
        md(f"""## 19. Model Training
Learning-rate search with the full 100-epoch budget for each of the 5 candidate values (same budget for every
optimizer); selection by **final validation Macro F1** (ties: lower validation objective). The test set is not used."""),
        code(f'''
OPT = "{opt}"
best_lr, lr_df, lr_hists = lr_search(d, OPT)
lr_df.to_csv(os.path.join(RESULTS, f"lr_search_{{OPT}}.csv"), index=False)
display(lr_df.round(4))
print("selected learning rate:", best_lr, "| value found by src/experiment_runner.py:", {selected_lr!r},
      "| match:", best_lr == {selected_lr!r})
lr_search_plot(lr_df, OPT, "{pre}_lr_search.png")
fig, ax = plt.subplots(figsize=(7, 3.6))
for lr, h in lr_hists.items():
    if not h["diverged"]:
        ax.plot(h["val_f1"], label=f"lr = {{lr:g}}")
ax.set_xlabel("epoch"); ax.set_ylabel("validation Macro F1"); ax.set_title(OPT + ": validation Macro F1 per learning rate"); ax.legend(fontsize=8)
save(fig, "{pre}_lr_curves.png")
'''),
        code(f'''
res = fit_and_evaluate(d, OPT, best_lr)        # final training run with the selected learning rate
h = res["history"]
print(f"epochs = {{len(h['train_obj']) - 1}} | parameter updates = {{h['updates'][-1]:,}} | samples processed = {{h['samples'][-1]:,}}")
print(f"training time = {{h['train_time'][-1]:.2f}} s | final J = {{res['obj_train']['total']:.4f}} "
      f"(hinge {{res['obj_train']['hinge']:.4f}} + RKHS reg {{res['obj_train']['reg']:.4f}})")
hist_df = history_frame(h); hist_df.to_csv(os.path.join(RESULTS, "training_histories", f"{{OPT}}.csv"), index=False)
display(hist_df.iloc[[0, 1, 2, 5, 10, 25, 50, 75, 100]].round(4))
'''),
    ]


def eval_cells(opt, pre):
    return [
        md("""## 20. Model Evaluation
Metrics on the **independent test set** (9 unseen subjects), computed manually from the confusion matrix and
verified against scikit-learn. The training objective (hinge + RKHS regularisation) is reported separately from the
evaluation metrics. ROC-AUC is One-vs-Rest from the raw decision scores (rank-based, no probabilities assumed);
no log-loss is reported because SVM scores are not calibrated probabilities."""),
        code(f'''
t = res["test"]
summary = pd.DataFrame([summary_row(OPT, res)]).T.rename(columns={{0: OPT}})
display(summary)
print("max |manual - sklearn| metric difference:", verify_with_sklearn(d["y_te"], np.argmax(decision_scores(d["H_te"], res["A"], res["b"]), 1), t))
report = classification_report_df(t, ACTIVITY_LABELS); display(report)
report.to_csv(os.path.join(RESULTS, "metrics", f"{{OPT}}_classification_report.csv"))
json.dump({{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in t.items()}},
          open(os.path.join(RESULTS, "metrics", f"{{OPT}}_test_metrics.json"), "w"), indent=1)
confusion(t["confusion_matrix"], ACTIVITY_LABELS, f"{{OPT}} - test confusion matrix", "{pre}_confusion_matrix.png")
'''),
        code(f'''
# Save the model with everything needed for inference (basis, gamma, scaler, labels), reload and re-predict
model = build_model(d, res)
path = os.path.join(MODELS, MODEL_FILES[OPT])
model.save(path, optimizer=OPT, learning_rate=best_lr, basis_idx=d["basis_idx"])
loaded = KernelMulticlassSVM.load(path)
pred_loaded = loaded.predict(raw["X_test"].to_numpy(float))
print("saved:", path, f"({{os.path.getsize(path) / 1e6:.1f}} MB)")
print("reloaded model reproduces the test predictions:", np.array_equal(pred_loaded, np.argmax(decision_scores(d["H_te"], res["A"], res["b"]), 1)))
'''),
        md("## 21. Visualizations"),
        code(f'''
training_curves({{OPT: h}}, keys=("train_obj", "val_obj", "train_acc", "val_acc", "val_f1"), name="{pre}_training_curves.png")
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(h["train_hinge"], label="hinge loss (train)"); ax.plot(h["train_reg"], label="RKHS regularisation")
ax.plot(h["train_obj"], label="total objective J", lw=2); ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.legend()
ax.set_title(OPT + ": objective decomposition"); save(fig, "{pre}_objective_parts.png")
per = pd.DataFrame({{"precision": t["per_class_precision"], "recall": t["per_class_recall"], "f1": t["per_class_f1"]}}, index=ACTIVITY_LABELS)
ax = per.plot.bar(figsize=(8, 3.6), rot=15, ylim=(0, 1.05), title=OPT + ": per-class test metrics"); save(ax.figure, "{pre}_per_class.png")
'''),
    ]


def common_head(opt):
    pre = PREFIX[opt]
    cells = [
        md(f"# 1. {T.TITLE}\n### Notebook {pre} of 5 - Optimizer: **{T.OPT_FULL[opt]}**\n"
           "Assignment 4 - Kernel-Based Multi-Class SVM implemented from scratch with NumPy."),
        md(T.ABSTRACT.format(opt=T.OPT_FULL[opt])), md(T.PROBLEM), md(T.OBJECTIVES), md(T.DATASET),
        md("""## 6. Library Imports
Only NumPy / pandas / Matplotlib are used for the model. scikit-learn is used **only** for
`GroupShuffleSplit`, for verifying the manual metrics, for ROC-AUC, and for the separate `SVC` reference baseline.
The project's helper modules are inlined below so the notebook runs stand-alone on Kaggle."""),
        code(SETUP),
        md("**Helper code - data loading** (`src/dataset_loader.py`)"), code(module_code("dataset_loader")),
        md("**Helper code - preprocessing** (`src/preprocessing.py`)"), code(module_code("preprocessing")),
        md("**Helper code - evaluation metrics** (`src/evaluation.py`)"), code(module_code("evaluation")),
        md("**Helper code - plotting** (`src/visualization.py`)"), code(module_code("visualization")),
        md("## 7. Dataset Loading"), code(EDA),
        md("## 8. Exploratory Data Analysis"), code(EDA2), code(EDA3),
        md("""**EDA observations.** No missing values; the six classes are reasonably balanced (largest/smallest
ratio about 1.4), so Macro F1 is a fair primary metric. Static postures (sitting, standing, lying) and dynamic
activities (walking variants) are clearly separated in the PCA plot, while **SITTING and STANDING overlap
strongly** - the hardest pair for any classifier. Many features are strongly correlated (they are derived from the
same signals)."""),
        md(T.PREPROC), code(PREP),
        md(T.THEORY_SVM), md(T.THEORY_KERNEL), md(T.THEORY_MULTI), md(T.THEORY_OBJECTIVE), md(T.OPT_THEORY[opt]),
        md("## 15. Kernel Implementation\nAll kernels are implemented manually (`src/kernels.py`). Squared distances use "
           "$\\lVert x\\rVert^2 + \\lVert z\\rVert^2 - 2x^\\top z$ with clipping of tiny negative values caused by round-off, "
           "and kernel matrices are evaluated in row batches."),
        code(module_code("kernels")), code(KERNEL_DEMO),
        md("**Explicit kernel approximations** (`src/kernel_approximation.py`) - used only for verification, not for the main comparison."),
        code(module_code("kernel_approximation")),
        md("## 16. Model Implementation\n`src/kernel_multiclass_svm.py`: scores $S = HA + b$, Crammer-Singer hinge loss, "
           "subgradient matrix $D$, RKHS regulariser $\\tfrac\\lambda2\\mathrm{Tr}(A^\\top G A)$ and the gradients "
           "$H^\\top D/N + \\lambda G A$ and $\\sum_i D_i/N$."),
        code(module_code("kernel_multiclass_svm")),
        md("## 17. Optimizer Implementation\n`src/optimizers.py`: the five update rules (written by hand) and the shared "
           "training loop. `src/experiment_runner.py`: data preparation, learning-rate search, evaluation helpers."),
        code(module_code("optimizers")), code(module_code("experiment_runner")),
        md("## 18. Hyperparameter Configuration"), code(CONFIG_CELL),
        md(MODE_MD.format(exact_ratio=(5551 / 2000) ** 2)), code(PREPARE),
    ]
    return cells


# ----------------------------------------------------------------------------- notebook-specific extras
EXTRA = {}

EXTRA["GD_pre"] = [
    md("""### Kernel hyperparameter selection (shared by all five notebooks)
Grid: $\\gamma \\in \\{0.5, 1, 2, 4, 8\\}\\times\\gamma_0$ (median heuristic) and $\\lambda \\in \\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\\}$.
Every grid point is trained with the **same fixed solver** (Adam, lr $10^{-3}$, 30 epochs, batch 64) on the training
split and scored by validation Macro F1. The winner is frozen in `CONFIG` **before** the optimizer comparison.
(Limitation: the selection solver is one of the five optimizers; the chosen $\\gamma,\\lambda$ are properties of the
model, and every optimizer then gets its own learning-rate search.)"""),
    code(r'''
sel_df, gm_best, lam_best = select_kernel_config(d)
sel_df.to_csv(os.path.join(RESULTS, "kernel_selection.csv"), index=False)
display(sel_df.pivot(index="gamma_multiplier", columns="lam", values="val_macro_f1").round(4))
print("best gamma multiplier:", gm_best, "| best lambda:", lam_best,
      "| identical to CONFIG:", gm_best == CONFIG["gamma_multiplier"] and lam_best == CONFIG["lam"])
sensitivity_lines(sel_df, "gamma_multiplier", "val_macro_f1", "lam", "Gamma sensitivity (validation)",
                  "01_gamma_sensitivity.png", xlabel="gamma / gamma0 (median heuristic)")
sensitivity_lines(sel_df, "lam", "val_macro_f1", "gamma_multiplier", "Regularisation sensitivity (validation)",
                  "01_lambda_sensitivity.png", xlabel="lambda")
'''),
    code(r'''
# Kernel Gram matrix of 300 basis samples (sorted by class) and RBF similarity curves
sub = np.random.default_rng(SEED).choice(len(d["Z"]), 300, replace=False)
gram_heatmap(d["G"][np.ix_(sub, sub)], d["y_tr"][d["basis_idx"]][sub], ACTIVITY_LABELS,
             "RBF Gram matrix (300 basis samples, sorted by class)", "01_gram_matrix_rbf.png")
d2 = squared_distances(d["X_tr"][:500], d["X_tr"][:500])
rbf_similarity_curves(d["gamma0"], [0.5, 1, 4, 8], np.percentile(d2, 95), "01_rbf_similarity.png")
# Why is GD slow here?  Conditioning of the coefficient-space problem:
ev = np.linalg.eigvalsh(d["H_tr"].T @ d["H_tr"] / len(d["H_tr"]))[::-1]
print("largest eigenvalues of H^T H / N:", np.round(ev[:5], 3), "| median eigenvalue: %.2e" % np.median(ev))
print("condition number lambda_max / lambda_median = %.1e -> progress along the many small-eigenvalue directions is very slow" % (ev[0] / np.median(ev)))
print("(the hinge loss is non-smooth: too large steps do not converge but make the subgradient iterates oscillate - see lr = 3 above)")
'''),
]

EXTRA["SGD_pre"] = [
    md("""### Verification of the efficient SGD implementation
The rotated-coordinate SGD must give exactly the same parameters as the naive SGD (which recomputes
$\\lambda G A$ at every single-sample update). We check this on a 400-sample subset for 2 epochs."""),
    code(r'''
sub = np.arange(400)
Hs, ys, Gs = d["H_tr"][sub][:, sub[:300]], d["y_tr"][sub], d["G"][:300, :300]
A_fast, b_fast, _ = train(Hs, ys, Gs, CONFIG["lam"], "SGD", 0.01, epochs=2, fast_sgd=True)
A_naive, b_naive, _ = train(Hs, ys, Gs, CONFIG["lam"], "SGD", 0.01, epochs=2, fast_sgd=False)
print("max |A_fast - A_naive| = %.2e, max |b_fast - b_naive| = %.2e" % (np.abs(A_fast - A_naive).max(), np.abs(b_fast - b_naive).max()))
'''),
    md("""### Kernel basis size analysis
Validation performance and cost versus the number of basis samples $M$ (the last row is the **exact** full basis,
$M = N$, Mode A). The fixed selection solver (Adam, lr $10^{-3}$, 30 epochs) and the shared $\\gamma, \\lambda$ are used."""),
    code(r'''
bs_df = basis_size_analysis()
bs_df.to_csv(os.path.join(RESULTS, "basis_size_analysis.csv"), index=False)
display(bs_df.round(4))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(bs_df.basis_M, bs_df.val_macro_f1, "o-"); ax[0].set_xlabel("basis size M"); ax[0].set_ylabel("validation Macro F1")
ax[0].set_title("Basis size vs performance")
ax[1].plot(bs_df.basis_M, bs_df.train_time_s, "o-", label="training (30 epochs)"); ax[1].plot(bs_df.basis_M, bs_df.kernel_time_s, "s-", label="kernel matrices")
ax[1].set_xlabel("basis size M"); ax[1].set_ylabel("seconds"); ax[1].set_title("Runtime vs basis size"); ax[1].legend()
save(fig, "02_basis_size_analysis.png")
'''),
]

EXTRA["SGD_post"] = [
    md("""### Noisy convergence inside one epoch
SGD updates after every sample. Below, the full training objective is recorded every 100 single-sample updates
during the first epoch (same initialisation, same shuffling seed) to visualise the noise."""),
    code(r'''
Lam, U = np.linalg.eigh(d["G"]); Ht = d["H_tr"] @ U
C = np.zeros((len(Lam), 6)); bb = np.zeros(6)
shrink = (1 - best_lr * CONFIG["lam"] * Lam)[:, None]
order = np.random.default_rng(CONFIG["seed"]).permutation(len(d["y_tr"]))
trace = [objective(U @ C, bb, d["H_tr"], d["y_tr"], d["G"], CONFIG["lam"])["total"]]
for s in range(0, len(order), 100):
    C, bb = sgd_epoch_rotated(Ht, d["y_tr"], C, bb, shrink, best_lr, order[s:s + 100])
    trace.append(objective(U @ C, bb, d["H_tr"], d["y_tr"], d["G"], CONFIG["lam"])["total"])
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(np.arange(len(trace)) * 100, trace, color=COLORS["SGD"]); ax.set_xlabel("single-sample updates (epoch 1)")
ax.set_ylabel("training objective J"); ax.set_title("SGD: objective during the first epoch"); save(fig, "02_sgd_within_epoch.png")
inc = np.mean(np.diff(trace) > 0)
print("share of 100-update blocks in which J increased: %.1f%%" % (100 * inc))
'''),
]

EXTRA["MBGD_post"] = [
    md("""### Batch-size study
Mini-Batch GD with the selected learning rate and batch sizes 16, 64 (default) and 256 (100 epochs each)."""),
    code(r'''
rows = []
for bs in (16, 64, 256):
    A_, b_, h_ = train(d["H_tr"], d["y_tr"], d["G"], CONFIG["lam"], "MBGD", best_lr, CONFIG["epochs"], batch_size=bs,
                       seed=CONFIG["seed"], H_val=d["H_va"], y_val=d["y_va"])
    rows.append({"batch_size": bs, "updates": h_["updates"][-1], "final_train_obj": h_["train_obj"][-1],
                 "val_macro_f1": h_["val_f1"][-1], "train_time_s": h_["train_time"][-1]})
batch_df = pd.DataFrame(rows); batch_df.to_csv(os.path.join(RESULTS, "batch_size_study.csv"), index=False)
display(batch_df.round(4))
'''),
    md("""### Supplementary kernel comparison (separate from the optimizer comparison)
Linear, polynomial (d = 2, 3; $\\gamma = 1/561$, $r = 1$), RBF and - exploratory only - sigmoid kernels on the
**same basis**, same $\\lambda$, same solver (Adam, lr $10^{-3}$, batch 64, 100 epochs). We also report the smallest
eigenvalue of each basis Gram matrix as a PSD check."""),
    code(r'''
kc = kernel_comparison(d)
kc.to_csv(os.path.join(RESULTS, "kernel_comparison.csv"), index=False)
display(kc.round(4))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(kc.kernel, kc.test_macro_f1, color=["#999", "#8c564b", "#bcbd22", "#d62728", "#17becf"])
for i, v in enumerate(kc.test_macro_f1): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel("test Macro F1"); ax.set_title("Kernel comparison (same basis, solver and budget)")
plt.setp(ax.get_xticklabels(), rotation=12, fontsize=8); save(fig, "03_kernel_comparison.png")
sub = np.random.default_rng(SEED).choice(len(d["Z"]), 200, replace=False)
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
yb = d["y_tr"][d["basis_idx"]][sub]; order_ = np.argsort(yb, kind="stable"); Zs = d["Z"][sub][order_]
for ax, (nm, kf) in zip(axes, [("linear", make_kernel("linear")), ("polynomial d=2", make_kernel("polynomial", degree=2)),
                               ("RBF", make_kernel("rbf", gamma=d["gamma"])), ("sigmoid", make_kernel("sigmoid"))]):
    Kk = kf(Zs, Zs); im = ax.imshow(Kk, cmap="viridis"); ax.set_title(f"{nm}\nmin eig = {np.linalg.eigvalsh(Kk)[0]:.2e}", fontsize=9)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Gram matrices of 200 basis samples sorted by class (block structure = within-class similarity)", fontsize=10)
save(fig, "03_gram_matrices_kernels.png")
'''),
    md("""### Synthetic demonstration: linear vs non-linear decision boundaries
A **synthetic** 2-D, 3-class dataset of concentric rings (clearly *not* HAR data). The same code trains a Linear
SVM (explicit features, $G = I$) and an RBF kernel SVM (Adam, 300 epochs)."""),
    code(r'''
rng_s = np.random.default_rng(SEED)
ys_ = rng_s.integers(0, 3, 600); ang = rng_s.uniform(0, 2 * np.pi, 600); rad = 1 + 1.5 * ys_ + rng_s.normal(0, 0.18, 600)
Xs_ = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
Gs_ = rbf_kernel(Xs_, Xs_, 1.0)
A_r, b_r, _ = train(Gs_, ys_, Gs_, 1e-3, "Adam", 0.05, 300, batch_size=32, n_classes=3)
W_l, b_l, _ = train(Xs_, ys_, None, 1e-3, "Adam", 0.05, 300, batch_size=32, n_classes=3)
acc_lin = np.mean(np.argmax(Xs_ @ W_l + b_l, 1) == ys_); acc_rbf = np.mean(np.argmax(Gs_ @ A_r + b_r, 1) == ys_)
decision_boundaries([lambda P: np.argmax(P @ W_l + b_l, 1), lambda P: np.argmax(rbf_kernel(P, Xs_, 1.0) @ A_r + b_r, 1)],
                    Xs_, ys_, [f"Linear SVM (train acc {acc_lin:.2f})", f"RBF kernel SVM (train acc {acc_rbf:.2f})"],
                    "03_synthetic_decision_boundaries.png")
'''),
]

EXTRA["AdaGrad_post"] = [
    code(r'''
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.semilogy(range(1, len(h["eff_lr"])), h["eff_lr"][1:], color=COLORS["AdaGrad"])
ax.set_xlabel("epoch"); ax.set_ylabel("median effective step  lr / (sqrt(acc) + eps)")
ax.set_title("AdaGrad: effective learning rate of A shrinks monotonically"); save(fig, "04_adagrad_effective_lr.png")
print("effective lr: epoch 1 = %.3e, epoch 100 = %.3e (base lr %g)" % (h["eff_lr"][1], h["eff_lr"][-1], best_lr))
'''),
    md(r"""### Linear SVM vs Kernel SVM (controlled comparison)
The previous Linear SVM assignment files were not available in this environment (its GitHub repository could not be
opened from the sandbox), so a **controlled Linear SVM reference** is trained here: the same Crammer-Singer hinge
loss with weights $W\in\mathbb R^{561\times 6}$ and penalty $\tfrac\lambda2\lVert W\rVert^2$ (= linear-kernel RKHS norm),
same data split, scaling, 100 epochs and Adam optimizer; $\lambda$ and then the learning rate are chosen on validation.
The **Kernel SVM** column is the RBF model trained with the optimizer that achieved the best validation Macro F1
in the five-optimizer comparison (notebook 05): **{best_opt}** (lr = {best_lr})."""),
    code(r'''
lin = linear_svm_reference(d, verbose=False)
display(lin["search"].round(4))
BEST_OPT, BEST_LR = "{best_opt}", {best_lr}
ker = fit_and_evaluate(d, BEST_OPT, BEST_LR)
svc = sklearn_svc_baseline(d, verbose=False)
display(svc["search"])
mem_lin = (lin["A"].nbytes + lin["b"].nbytes) / 2**20
mem_ker = (ker["A"].nbytes + ker["b"].nbytes + d["Z"].nbytes) / 2**20
lvk = pd.DataFrame({
    "Linear SVM": [lin["test"]["accuracy"], lin["test"]["macro_precision"], lin["test"]["macro_recall"], lin["test"]["macro_f1"],
                   lin["train_time"], lin["pred_time"], f"{lin['A'].size + 6:,} weights", "Linear", f"{mem_lin:.3f} MiB"],
    f"Kernel SVM (RBF, {BEST_OPT})": [ker["test"]["accuracy"], ker["test"]["macro_precision"], ker["test"]["macro_recall"], ker["test"]["macro_f1"],
                   ker["history"]["train_time"][-1], ker["pred_time_with_kernel"], f"{ker['A'].size + 6:,} coefficients + 2,000 x 561 basis", "Non-linear", f"{mem_ker:.2f} MiB"],
    "sklearn SVC (RBF, OvO, reference)": [svc["test"]["accuracy"], svc["test"]["macro_precision"], svc["test"]["macro_recall"], svc["test"]["macro_f1"],
                   svc["train_time"], svc["pred_time"], f"{svc['n_support']:,} support vectors", "Non-linear", "-"]},
    index=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Training Time (s)", "Prediction Time (s)",
           "Model Complexity", "Decision Boundary", "Model Memory"])
lvk.to_csv(os.path.join(RESULTS, "linear_vs_kernel.csv"))
display(lvk)
print("Linear SVM: lambda = %g, lr = %g | SVC: C = %g, gamma = %.4e" % (lin["lam"], lin["lr"], svc["C"], d["gamma"]))
json.dump({"linear_svm": {"lam": lin["lam"], "lr": lin["lr"]}, "svc_C": svc["C"], "kernel_svm_optimizer": BEST_OPT, "kernel_svm_lr": BEST_LR},
          open(os.path.join(RESULTS, "baselines.json"), "w"), indent=2)
print("Note: kernel prediction time includes evaluating k(x_test, basis); the linear model needs only X W + b.")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
confusion(lin["test"]["confusion_matrix"], ACTIVITY_LABELS, "Linear SVM (test)", None, ax=axes[0])
confusion(ker["test"]["confusion_matrix"], ACTIVITY_LABELS, f"RBF Kernel SVM - {BEST_OPT} (test)", None, ax=axes[1])
save(fig, "04_linear_vs_kernel_confusion.png")
'''),
    md("""### Exact-kernel verification and approximation quality
(1) Nystroem and Random Fourier Features: relative Frobenius error $\\lVert K - \\Phi\\Phi^\\top\\rVert_F/\\lVert K\\rVert_F$ on 1,000
validation samples. (2) Small exact demonstration on 1,500 training samples: exact full basis (Mode A) vs restricted
basis of 300 (Mode B) vs Nystroem features with the same 300 landmarks (Mode C)."""),
    code(r'''
aq = approximation_check(d); aq.to_csv(os.path.join(RESULTS, "approximation_quality.csv"), index=False)
display(aq.pivot(index="dimension", columns="method", values="rel_frobenius_error").round(4))
ed = exact_kernel_demo(d); ed.to_csv(os.path.join(RESULTS, "exact_kernel_demo.csv"), index=False)
display(ed.round(4))
'''),
]

EXTRA["Adam_post"] = [
    md("""## Final optimizer comparison (all five optimizers)
To keep this notebook stand-alone, the four other optimizers are re-trained here with the learning rates selected
in their own notebooks (embedded below and identical to `results/hyperparameters.json`); training is deterministic,
so the metrics reproduce those notebooks exactly (training times vary slightly between runs)."""),
    code(r'''
SELECTED_LR = {selected}
all_res = {}
for o in ("GD", "SGD", "MBGD", "AdaGrad"):
    all_res[o] = fit_and_evaluate(d, o, SELECTED_LR[o]); print(o, "done", round(all_res[o]["history"]["train_time"][-1], 1), "s")
all_res["Adam"] = res
assert best_lr == SELECTED_LR["Adam"]
comp = pd.DataFrame([summary_row(o, r) for o, r in all_res.items()])
comp.to_csv(os.path.join(RESULTS, "optimizer_comparison.csv"), index=False)
for o, r in all_res.items():
    history_frame(r["history"]).to_csv(os.path.join(RESULTS, "training_histories", f"{o}.csv"), index=False)
table = comp[["optimizer", "accuracy", "macro_precision", "macro_recall", "macro_f1", "train_hinge", "regularization",
              "train_objective", "training_time_s", "prediction_time_s"]].rename(columns={
    "optimizer": "Optimizer", "accuracy": "Accuracy", "macro_precision": "Macro Precision", "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1", "train_hinge": "Hinge Loss", "regularization": "Regularization Loss", "train_objective": "Total Objective",
    "training_time_s": "Training Time (s)", "prediction_time_s": "Prediction Time (s)"})
display(table.round(4))
display(comp[["optimizer", "learning_rate", "batch_size", "epochs", "updates", "samples_processed", "val_macro_f1",
              "val_accuracy", "roc_auc_ovr", "pct_epochs_obj_increased", "tail_rel_std", "prediction_time_with_kernel_s"]].round(4))
'''),
    code(r'''
H = {o: r["history"] for o, r in all_res.items()}
training_curves(H, keys=("train_obj", "val_obj"), name="05_loss_curves_all.png")
training_curves(H, keys=("train_acc", "val_acc", "val_f1"), name="05_accuracy_f1_curves_all.png")
fig, ax = plt.subplots(figsize=(7, 3.6))
for o, hh in H.items():
    ax.plot(np.maximum(hh["updates"], 1), hh["train_obj"], color=COLORS[o], label=NAMES[o])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("parameter updates (log)"); ax.set_ylabel("training objective J")
ax.set_title("Convergence per parameter update (an epoch is not an update)"); ax.legend(fontsize=8); save(fig, "05_convergence_vs_updates.png")
bar_compare({o: r["history"]["train_time"][-1] for o, r in all_res.items()}, "seconds", "Training time (100 epochs)", "05_training_time.png", "{:.1f}")
bar_compare({o: 1000 * r["pred_time"] for o, r in all_res.items()}, "milliseconds", "Prediction time on 2,947 test windows (cached kernel)", "05_prediction_time.png", "{:.2f}")
bar_compare({o: r["test"]["macro_f1"] for o, r in all_res.items()}, "test Macro F1", "Final test Macro F1", "05_final_macro_f1.png", "{:.4f}")
confusion_grid({o: r["test"]["confusion_matrix"] for o, r in all_res.items()}, ACTIVITY_LABELS, "05_confusion_matrices_all.png")
per_class_f1({o: r["test"]["per_class_f1"] for o, r in all_res.items()}, ACTIVITY_LABELS, "05_per_class_f1.png")
pc = pd.DataFrame({o: r["test"]["per_class_f1"] for o, r in all_res.items()}, index=ACTIVITY_LABELS)
pc.to_csv(os.path.join(RESULTS, "per_class_f1.csv")); display(pc.round(4))
'''),
    code(r'''
# Automatic determination of the "winners" (evidence from the table above)
target = 1.10 * comp.train_objective.min()
def epochs_to(hh, thr):
    hit = np.flatnonzero(np.asarray(hh["train_obj"]) <= thr)
    return int(hit[0]) if len(hit) else None
conv = {o: epochs_to(r["history"], target) for o, r in all_res.items()}
# efficiency: the fastest optimizer among those within 1 Macro-F1 point of the best test Macro F1
near_best = comp[comp.macro_f1 >= comp.macro_f1.max() - 0.01]
eff_winner = near_best.loc[near_best.training_time_s.idxmin(), "optimizer"]
best_class = {c: pc.loc[c].idxmax() for c in ACTIVITY_LABELS}
verdict = {
    "best_by_validation_macro_f1": comp.loc[comp.val_macro_f1.idxmax(), "optimizer"],
    "highest_test_accuracy": comp.loc[comp.accuracy.idxmax(), "optimizer"],
    "lowest_training_objective": comp.loc[comp.train_objective.idxmin(), "optimizer"],
    "fastest_training": comp.loc[comp.training_time_s.idxmin(), "optimizer"],
    "convergence_target_J": round(float(target), 4), "epochs_to_target": conv,
    "fastest_convergence_epochs": min((v, o) for o, v in conv.items() if v is not None)[1],
    "most_stable_epoch_level (lowest % epochs with J increase, then tail std)": comp.sort_values(["pct_epochs_obj_increased", "tail_rel_std"]).iloc[0]["optimizer"],
    "best_efficiency (fastest within 1 Macro-F1 point of the best)": eff_winner,
    "best_per_class_f1": best_class}
json.dump(verdict, open(os.path.join(RESULTS, "optimizer_verdict.json"), "w"), indent=2)
print(json.dumps(verdict, indent=2))
'''),
]


def build(opt, selected, best_opt):
    pre = PREFIX[opt]
    cells = common_head(opt)
    cells += EXTRA.get(f"{opt}_pre", [])
    cells += train_cells(opt, pre, selected[opt])
    cells += eval_cells(opt, pre)
    extra = EXTRA.get(f"{opt}_post", [])
    for c in extra:
        c = copy.deepcopy(c)
        c.source = c.source.replace("{best_opt}", best_opt).replace("{best_lr}", repr(selected[best_opt])).replace(
            "{selected}", repr(selected))
        cells.append(c)
    disc = discussion(opt)
    cells += [md("## 22. Results Interpretation\n" + disc["interpretation"]), md("## 23. Conclusion\n" + disc["conclusion"]),
              md(T.REFERENCES)]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    path = os.path.join(ROOT, "notebooks", NB_FILES[opt])
    nbf.write(nb, path)
    print("wrote", path, len(cells), "cells")


if __name__ == "__main__":
    hp = json.load(open(os.path.join(ROOT, "results", "hyperparameters.json")))
    comp = __import__("pandas").read_csv(os.path.join(ROOT, "results", "optimizer_comparison.csv"))
    best_opt = comp.loc[comp.val_macro_f1.idxmax(), "optimizer"]
    only = sys.argv[1:] or list(NB_FILES)
    for o in only:
        build(o, hp["selected_lr"], best_opt)
