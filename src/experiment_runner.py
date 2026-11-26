"""Shared experiment pipeline: data preparation, resource report, learning-rate search,
final training/evaluation and the supplementary experiments.

Run everything from the project root with:   python src/experiment_runner.py
"""
import json
import os
import platform
import time
import numpy as np
import pandas as pd

from dataset_loader import load_har, ACTIVITY_LABELS  # local-import
from preprocessing import (encode_labels, subject_group_split, Standardizer, select_basis,  # local-import
                           median_heuristic_gamma, SEED)  # local-import
from kernels import rbf_kernel, kernel_matrix, make_kernel  # local-import
from kernel_multiclass_svm import objective, decision_scores, KernelMulticlassSVM  # local-import
from optimizers import train, DEFAULT_BATCH  # local-import
from evaluation import evaluate  # local-import

# ----------------------------------------------------------------------------- shared configuration
# Selected ONCE on the validation split (train + validation only, see select_kernel_config) and then
# frozen for the main five-optimizer comparison.
CONFIG = {
    "seed": SEED,
    "val_fraction": 0.2,
    "kernel": "rbf",
    "n_basis": 2000,                 # Mode B: reduced, class-stratified kernel basis
    "gamma_multiplier": 4.0,         # gamma = multiplier * median-heuristic gamma (selected on validation)
    "lam": 1e-4,                     # selected on validation
    "epochs": 100,
    "batch_size": 64,
}
LR_GRID = {
    "GD": [0.03, 0.1, 0.3, 1.0, 3.0],
    "SGD": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "MBGD": [3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "AdaGrad": [3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "Adam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}
N_CLASSES = len(ACTIVITY_LABELS)
MODEL_FILES = {"GD": "kernel_svm_gd.npz", "SGD": "kernel_svm_sgd.npz", "MBGD": "kernel_svm_minibatch.npz",
               "AdaGrad": "kernel_svm_adagrad.npz", "Adam": "kernel_svm_adam.npz"}


def find_project_root():
    p = os.getcwd()
    for cand in (p, os.path.dirname(p)):
        if os.path.isdir(os.path.join(cand, "src")) and os.path.isdir(os.path.join(cand, "results")):
            return cand
    return p


def resource_report(n_train, n_features, n_basis, n_val, n_test, n_classes=N_CLASSES, itemsize=8):
    """Estimate memory / compute for exact (Mode A) and reduced (Mode B) kernel representations."""
    try:
        import psutil
        avail = psutil.virtual_memory().available / 2**30
    except Exception:
        avail = float("nan")
    gib = lambda n: n * itemsize / 2**30
    rows = []
    for mode, M in (("A (exact, M = N)", n_train), (f"B (reduced, M = {n_basis})", n_basis)):
        mem = gib(n_train * M + M * M + (n_val + n_test) * M)
        rows.append({"mode": mode, "basis_M": M,
                     "H_train_GiB": round(gib(n_train * M), 3), "G_basis_GiB": round(gib(M * M), 3),
                     "H_val+test_GiB": round(gib((n_val + n_test) * M), 3), "total_GiB": round(mem, 3),
                     "kernel_eval_GFLOP": round((n_train + n_val + n_test + M) * M * n_features * 2 / 1e9, 2),
                     "GD_iter_GFLOP": round(2 * 2 * n_train * M * n_classes / 1e9 + 2 * M * M * n_classes / 1e9, 3),
                     "adaptive_update_regGFLOP": round(2 * M * M * n_classes / 1e9, 4)})
    info = {"cpu_count": os.cpu_count(), "available_RAM_GiB": round(avail, 2),
            "python": platform.python_version(), "numpy": np.__version__}
    return pd.DataFrame(rows), info


def prepare_data(n_basis=None, gamma_multiplier=None, kernel="rbf", kernel_params=None, seed=SEED,
                 verbose=True, build_kernels=True):
    """Load, split by subject, standardise, choose basis, compute gamma and the cached kernel matrices."""
    n_basis = CONFIG["n_basis"] if n_basis is None else n_basis
    gamma_multiplier = CONFIG["gamma_multiplier"] if gamma_multiplier is None else gamma_multiplier
    raw = load_har()
    y_full = encode_labels(raw["y_train"])
    y_test = encode_labels(raw["y_test"])
    subj = raw["subject_train"].to_numpy()
    tr_idx, va_idx = subject_group_split(subj, CONFIG["val_fraction"], seed)
    X_full = raw["X_train"].to_numpy(float)                   # subject ids are NOT features
    scaler = Standardizer().fit(X_full[tr_idx])               # fitted on the training split only
    X_tr, X_va = scaler.transform(X_full[tr_idx]), scaler.transform(X_full[va_idx])
    X_te = scaler.transform(raw["X_test"].to_numpy(float))
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]
    basis_idx = select_basis(y_tr, n_basis, seed)              # basis taken from the training split only
    Z = X_tr[basis_idx]
    gamma0 = median_heuristic_gamma(X_tr, seed=seed)
    gamma = gamma0 * gamma_multiplier
    d = dict(raw=raw, X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_test,
             tr_idx=tr_idx, va_idx=va_idx, subjects_tr=np.unique(subj[tr_idx]), subjects_va=np.unique(subj[va_idx]),
             scaler=scaler, basis_idx=basis_idx, Z=Z, gamma0=gamma0, gamma=gamma, kernel=kernel)
    if build_kernels:
        params = {"gamma": gamma} if kernel == "rbf" else dict(kernel_params or {})
        kfun = make_kernel(kernel, **params)
        t0 = time.perf_counter()
        d["H_tr"] = kernel_matrix(X_tr, Z, kfun)
        d["H_va"] = kernel_matrix(X_va, Z, kfun)
        d["H_te"] = kernel_matrix(X_te, Z, kfun)
        d["G"] = kfun(Z, Z)
        d["G"] = (d["G"] + d["G"].T) / 2.0
        d["kernel_params"] = params
        d["kernel_time"] = time.perf_counter() - t0
    if verbose:
        print(f"train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}  basis M={len(basis_idx)}  "
              f"gamma0={gamma0:.3e}  gamma={gamma:.3e}")
    return d


def lr_search(d, optimizer, grid=None, epochs=None, lam=None, batch_size=None, verbose=True):
    """Equal tuning budget: every optimizer tries 5 learning rates with the full epoch budget.
    Selection criterion: final-epoch VALIDATION Macro F1 (ties -> lower validation objective)."""
    grid = LR_GRID[optimizer] if grid is None else grid
    epochs = CONFIG["epochs"] if epochs is None else epochs
    lam = CONFIG["lam"] if lam is None else lam
    rows, hists = [], {}
    for lr in grid:
        A, b, h = train(d["H_tr"], d["y_tr"], d["G"], lam, optimizer, lr, epochs,
                        batch_size=batch_size or DEFAULT_BATCH[optimizer], seed=CONFIG["seed"],
                        H_val=d["H_va"], y_val=d["y_va"])
        ok = not h["diverged"]
        rows.append({"optimizer": optimizer, "lr": lr, "diverged": h["diverged"],
                     "final_train_obj": h["train_obj"][-1] if ok else np.nan,
                     "final_val_obj": h["val_obj"][-1] if ok else np.nan,
                     "final_val_acc": h["val_acc"][-1] if ok else np.nan,
                     "final_val_macro_f1": h["val_f1"][-1] if ok else np.nan,
                     "train_time_s": h["train_time"][-1]})
        hists[lr] = h
        if verbose:
            print(f"{optimizer:8s} lr={lr:<8g} " + ("DIVERGED" if not ok else
                  f"J={rows[-1]['final_train_obj']:.4f} valJ={rows[-1]['final_val_obj']:.4f} "
                  f"valF1={rows[-1]['final_val_macro_f1']:.4f}  ({rows[-1]['train_time_s']:.1f}s)"))
    df = pd.DataFrame(rows)
    ok = df[~df.diverged].sort_values(["final_val_macro_f1", "final_val_obj"], ascending=[False, True])
    return float(ok.iloc[0]["lr"]), df, hists


def fit_and_evaluate(d, optimizer, lr, lam=None, epochs=None, batch_size=None):
    """Train with the selected lr, then evaluate on train / validation / TEST (test used only here)."""
    lam = CONFIG["lam"] if lam is None else lam
    epochs = CONFIG["epochs"] if epochs is None else epochs
    bs = batch_size or DEFAULT_BATCH[optimizer]
    A, b, h = train(d["H_tr"], d["y_tr"], d["G"], lam, optimizer, lr, epochs, batch_size=bs,
                    seed=CONFIG["seed"], H_val=d["H_va"], y_val=d["y_va"])
    t0 = time.perf_counter()
    S_te = decision_scores(d["H_te"], A, b)
    pred_te = np.argmax(S_te, axis=1)
    pred_time = time.perf_counter() - t0
    # prediction time including the kernel evaluation k(x_test, basis) - the realistic inference cost
    t0 = time.perf_counter()
    _ = np.argmax(rbf_kernel(d["X_te"], d["Z"], d["gamma"]) @ A + b, axis=1)
    pred_time_with_kernel = time.perf_counter() - t0
    m_te = evaluate(d["y_te"], pred_te, N_CLASSES, scores=S_te)
    obj_tr = objective(A, b, d["H_tr"], d["y_tr"], d["G"], lam)
    obj_te = objective(A, b, d["H_te"], d["y_te"], d["G"], lam)
    m_va = evaluate(d["y_va"], np.argmax(decision_scores(d["H_va"], A, b), axis=1), N_CLASSES)
    return dict(A=A, b=b, history=h, test=m_te, val=m_va, obj_train=obj_tr, obj_test=obj_te,
                pred_time=pred_time, pred_time_with_kernel=pred_time_with_kernel, lr=lr, batch_size=bs,
                optimizer=optimizer)


def build_model(d, res, lam=None):
    m = KernelMulticlassSVM("rbf", {"gamma": d["gamma"]}, CONFIG["lam"] if lam is None else lam,
                            N_CLASSES, ACTIVITY_LABELS)
    m.A, m.b, m.basis = res["A"], res["b"], d["Z"]
    m.scaler_mean, m.scaler_scale = d["scaler"].mean_, d["scaler"].scale_
    return m


def history_frame(h):
    n = len(h["train_obj"])
    return pd.DataFrame({k: h[k][:n] for k in h if k != "diverged" and len(h[k]) >= n}).rename_axis("epoch").reset_index()


def stability_stats(h):
    obj = np.asarray(h["train_obj"])
    inc = np.diff(obj) > 1e-12
    tail = obj[-20:]
    return {"pct_epochs_obj_increased": float(100 * inc.mean()),
            "tail_rel_std": float(np.std(tail) / np.mean(tail))}


def summary_row(name, res, lam=None):
    h, t = res["history"], res["test"]
    return {"optimizer": name, "learning_rate": res["lr"], "batch_size": res["batch_size"],
            "epochs": len(h["train_obj"]) - 1, "updates": h["updates"][-1], "samples_processed": h["samples"][-1],
            "accuracy": t["accuracy"], "macro_precision": t["macro_precision"], "macro_recall": t["macro_recall"],
            "macro_f1": t["macro_f1"], "weighted_f1": t["weighted_f1"], "roc_auc_ovr": t["roc_auc_ovr_macro"],
            "val_macro_f1": res["val"]["macro_f1"], "val_accuracy": res["val"]["accuracy"],
            "train_hinge": res["obj_train"]["hinge"], "regularization": res["obj_train"]["reg"],
            "train_objective": res["obj_train"]["total"], "test_hinge": res["obj_test"]["hinge"],
            "training_time_s": h["train_time"][-1], "prediction_time_s": res["pred_time"],
            "prediction_time_with_kernel_s": res["pred_time_with_kernel"], **stability_stats(h)}


# ----------------------------------------------------------------------------- supplementary experiments
SELECTION = {"gamma_multipliers": [0.5, 1, 2, 4, 8], "lams": [1e-5, 1e-4, 1e-3, 1e-2],
             "solver": "Adam", "solver_lr": 1e-3, "solver_epochs": 30}


def select_kernel_config(base=None, verbose=True):
    """Grid search over RBF gamma (multiples of the median heuristic) and lambda on the VALIDATION split.

    A single fixed solver (Adam, lr 1e-3, 30 epochs, batch 64) is used for every grid point so that the
    kernel configuration is chosen before - and independently of - the five-optimizer comparison.
    Criterion: validation Macro F1 (ties -> lower validation objective).
    """
    rows = []
    base = base or prepare_data(build_kernels=False, verbose=False)
    for gm in SELECTION["gamma_multipliers"]:
        gamma = base["gamma0"] * gm
        H_tr = kernel_matrix(base["X_tr"], base["Z"], make_kernel("rbf", gamma=gamma))
        H_va = kernel_matrix(base["X_va"], base["Z"], make_kernel("rbf", gamma=gamma))
        G = rbf_kernel(base["Z"], base["Z"], gamma)
        for lam in SELECTION["lams"]:
            A, b, h = train(H_tr, base["y_tr"], G, lam, SELECTION["solver"], SELECTION["solver_lr"],
                            SELECTION["solver_epochs"], batch_size=64, seed=SEED, H_val=H_va, y_val=base["y_va"])
            rows.append({"gamma_multiplier": gm, "gamma": gamma, "lam": lam, "val_macro_f1": h["val_f1"][-1],
                         "val_accuracy": h["val_acc"][-1], "val_obj": h["val_obj"][-1], "train_obj": h["train_obj"][-1],
                         "train_macro_f1": h["train_f1"][-1]})
            if verbose:
                print(f"gamma x{gm:<4g} lam={lam:<7g} valF1={rows[-1]['val_macro_f1']:.4f}  trainF1={rows[-1]['train_macro_f1']:.4f}")
    df = pd.DataFrame(rows)
    best = df.sort_values(["val_macro_f1", "val_obj"], ascending=[False, True]).iloc[0]
    return df, float(best["gamma_multiplier"]), float(best["lam"])


def basis_size_analysis(sizes=(250, 500, 1000, 2000, 4000, None), verbose=True):
    """Validation performance and cost versus the number of basis samples M (None = all = exact Mode A)."""
    rows = []
    for M in sizes:
        d = prepare_data(n_basis=M if M is not None else 10**9, verbose=False)
        A, b, h = train(d["H_tr"], d["y_tr"], d["G"], CONFIG["lam"], SELECTION["solver"], SELECTION["solver_lr"],
                        SELECTION["solver_epochs"], batch_size=64, seed=SEED, H_val=d["H_va"], y_val=d["y_va"])
        Mreal = len(d["basis_idx"])
        rows.append({"basis_M": Mreal, "exact": M is None, "val_macro_f1": h["val_f1"][-1], "val_accuracy": h["val_acc"][-1],
                     "kernel_time_s": d["kernel_time"], "train_time_s": h["train_time"][-1],
                     "kernel_memory_MiB": (d["H_tr"].nbytes + d["H_va"].nbytes + d["H_te"].nbytes + d["G"].nbytes) / 2**20})
        if verbose:
            print(rows[-1])
        del d
    return pd.DataFrame(rows)


def kernel_comparison(d, epochs=None, lr=1e-3, verbose=True):
    """Supplementary: Linear / Polynomial / RBF (and exploratory Sigmoid) kernels on the SAME basis,
    same lambda, same solver (Adam, lr 1e-3, batch 64). Separate from the main optimizer comparison."""
    epochs = CONFIG["epochs"] if epochs is None else epochs
    nf = d["X_tr"].shape[1]
    specs = [("linear", {}), ("polynomial", {"gamma": 1.0 / nf, "degree": 2, "coef0": 1.0}),
             ("polynomial", {"gamma": 1.0 / nf, "degree": 3, "coef0": 1.0}), ("rbf", {"gamma": d["gamma"]}),
             ("sigmoid", {"gamma": 1.0 / nf, "coef0": 0.0})]
    rows = []
    for name, params in specs:
        k = make_kernel(name, **params)
        t0 = time.perf_counter()
        H_tr, H_va, H_te = (kernel_matrix(X, d["Z"], k) for X in (d["X_tr"], d["X_va"], d["X_te"]))
        G = k(d["Z"], d["Z"]); G = (G + G.T) / 2
        kt = time.perf_counter() - t0
        A, b, h = train(H_tr, d["y_tr"], G, CONFIG["lam"], "Adam", lr, epochs, batch_size=64, seed=SEED,
                        H_val=H_va, y_val=d["y_va"])
        S = decision_scores(H_te, A, b)
        m = evaluate(d["y_te"], np.argmax(S, 1), N_CLASSES, scores=S)
        label = name if name != "polynomial" else f"polynomial (d={params['degree']})"
        if name == "sigmoid":
            label += " [exploratory]"
        eig = np.linalg.eigvalsh(G)
        rows.append({"kernel": label, "params": json.dumps({k_: round(v, 6) for k_, v in params.items()}),
                     "min_eig_G": float(eig[0]), "psd_within_tol": bool(eig[0] > -1e-8 * abs(eig[-1])),
                     "diverged": h["diverged"], "val_macro_f1": h["val_f1"][-1], "test_accuracy": m["accuracy"],
                     "test_macro_f1": m["macro_f1"], "train_objective": h["train_obj"][-1],
                     "kernel_time_s": kt, "train_time_s": h["train_time"][-1]})
        if verbose:
            print({k_: rows[-1][k_] for k_ in ("kernel", "min_eig_G", "val_macro_f1", "test_macro_f1")})
    return pd.DataFrame(rows)


def linear_svm_reference(d, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), lrs=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2), verbose=True):
    """Controlled Linear multi-class SVM (same Crammer-Singer hinge loss, primal W, penalty (lam/2)||W||^2),
    same data / split / scaling / epochs, Adam optimiser. lambda then lr selected on validation."""
    rows = []
    for lam in lams:
        A, b, h = train(d["X_tr"], d["y_tr"], None, lam, "Adam", 1e-3, CONFIG["epochs"], 64, SEED,
                        H_val=d["X_va"], y_val=d["y_va"])
        rows.append({"stage": "lambda", "lam": lam, "lr": 1e-3, "val_macro_f1": h["val_f1"][-1], "val_obj": h["val_obj"][-1]})
    lam_best = pd.DataFrame(rows).sort_values(["val_macro_f1", "val_obj"], ascending=[False, True]).iloc[0]["lam"]
    for lr in lrs:
        A, b, h = train(d["X_tr"], d["y_tr"], None, lam_best, "Adam", lr, CONFIG["epochs"], 64, SEED,
                        H_val=d["X_va"], y_val=d["y_va"])
        rows.append({"stage": "lr", "lam": lam_best, "lr": lr, "val_macro_f1": h["val_f1"][-1], "val_obj": h["val_obj"][-1],
                     "diverged": h["diverged"]})
    search = pd.DataFrame(rows)
    sel = search[search.stage == "lr"].sort_values(["val_macro_f1", "val_obj"], ascending=[False, True]).iloc[0]
    A, b, h = train(d["X_tr"], d["y_tr"], None, sel["lam"], "Adam", sel["lr"], CONFIG["epochs"], 64, SEED,
                    H_val=d["X_va"], y_val=d["y_va"])
    t0 = time.perf_counter(); S = d["X_te"] @ A + b; pred = np.argmax(S, 1); pt = time.perf_counter() - t0
    m = evaluate(d["y_te"], pred, N_CLASSES, scores=S)
    if verbose:
        print(search.to_string())
    return dict(A=A, b=b, history=h, test=m, lam=float(sel["lam"]), lr=float(sel["lr"]), pred_time=pt,
                train_time=h["train_time"][-1], search=search, obj_train=objective(A, b, d["X_tr"], d["y_tr"], None, sel["lam"]))


def sklearn_svc_baseline(d, Cs=(0.1, 1, 10, 100), verbose=True):
    """Reference only: scikit-learn SVC (libsvm SMO solver, One-vs-One), same gamma, C chosen on validation."""
    from sklearn.svm import SVC
    rows = []
    for C in Cs:
        clf = SVC(kernel="rbf", C=C, gamma=d["gamma"]).fit(d["X_tr"], d["y_tr"])
        from sklearn.metrics import f1_score
        rows.append({"C": C, "val_macro_f1": f1_score(d["y_va"], clf.predict(d["X_va"]), average="macro"),
                     "n_support": int(clf.n_support_.sum())})
        if verbose:
            print(rows[-1])
    search = pd.DataFrame(rows)
    C = float(search.sort_values("val_macro_f1", ascending=False).iloc[0]["C"])
    t0 = time.perf_counter(); clf = SVC(kernel="rbf", C=C, gamma=d["gamma"]).fit(d["X_tr"], d["y_tr"]); tt = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = clf.predict(d["X_te"]); pt = time.perf_counter() - t0
    m = evaluate(d["y_te"], pred, N_CLASSES, scores=clf.decision_function(d["X_te"]))
    return dict(C=C, test=m, train_time=tt, pred_time=pt, n_support=int(clf.n_support_.sum()), search=search)


def approximation_check(d, n_eval=1000, seed=SEED):
    """Quality of Nystrom (same basis Z) and Random Fourier Features on a validation subset."""
    from kernel_approximation import Nystroem, RandomFourierFeatures, approximation_error  # local-import
    rng = np.random.default_rng(seed)
    Xs = d["X_va"][rng.choice(len(d["X_va"]), n_eval, replace=False)]
    rows = []
    for M in (250, 500, 1000, 2000):
        Z = d["Z"][rng.choice(len(d["Z"]), M, replace=False)] if M < len(d["Z"]) else d["Z"]
        ny = Nystroem(d["gamma"]).fit(Z)
        rows.append({"method": "Nystroem", "dimension": M, "rel_frobenius_error": approximation_error(Xs, d["gamma"], ny)})
        rff = RandomFourierFeatures(d["gamma"], M, seed).fit(d["X_tr"])
        rows.append({"method": "Random Fourier Features", "dimension": M, "rel_frobenius_error": approximation_error(Xs, d["gamma"], rff)})
    return pd.DataFrame(rows)


def exact_kernel_demo(d, n_sub=1500, m_small=300, epochs=60, seed=SEED):
    """Small EXACT-kernel verification on a subject-split subset: full basis (Mode A) vs restricted basis
    (Mode B) vs Nystroem features (Mode C), all trained with Adam on the same objective."""
    from kernel_approximation import Nystroem  # local-import
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(d["y_tr"]), n_sub, replace=False))
    X, y = d["X_tr"][idx], d["y_tr"][idx]
    Xv, yv = d["X_va"], d["y_va"]
    g, lam = d["gamma"], CONFIG["lam"]
    rows = []
    for mode, bidx in (("A: exact (M = N)", np.arange(n_sub)), (f"B: restricted (M = {m_small})", select_basis(y, m_small, seed))):
        Z = X[bidx]; H = rbf_kernel(X, Z, g); G = rbf_kernel(Z, Z, g); Hv = rbf_kernel(Xv, Z, g)
        A, b, h = train(H, y, G, lam, "Adam", 1e-3, epochs, 64, seed, H_val=Hv, y_val=yv)
        rows.append({"mode": mode, "basis_M": len(bidx), "final_train_obj": h["train_obj"][-1], "val_macro_f1": h["val_f1"][-1]})
    ny = Nystroem(g).fit(X[select_basis(y, m_small, seed)])
    P, Pv = ny.transform(X), ny.transform(Xv)
    A, b, h = train(P, y, None, lam, "Adam", 1e-3, epochs, 64, seed, H_val=Pv, y_val=yv)
    rows.append({"mode": f"C: Nystroem features (M = {m_small})", "basis_M": m_small, "final_train_obj": h["train_obj"][-1],
                 "val_macro_f1": h["val_f1"][-1]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- full pipeline (CLI)
def main():
    root = find_project_root()
    res_dir = os.path.join(root, "results"); os.makedirs(res_dir, exist_ok=True)
    for sub in ("training_histories", "metrics", "figures"):
        os.makedirs(os.path.join(res_dir, sub), exist_ok=True)
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    d = prepare_data()
    rep, info = resource_report(len(d["y_tr"]), d["X_tr"].shape[1], CONFIG["n_basis"], len(d["y_va"]), len(d["y_te"]))
    rep.to_csv(os.path.join(res_dir, "resource_report.csv"), index=False); print(rep.to_string(), info)
    sel, gm, lam = select_kernel_config(prepare_data(build_kernels=False, verbose=False))
    sel.to_csv(os.path.join(res_dir, "kernel_selection.csv"), index=False)
    assert gm == CONFIG["gamma_multiplier"] and lam == CONFIG["lam"], (gm, lam)
    basis_size_analysis().to_csv(os.path.join(res_dir, "basis_size_analysis.csv"), index=False)
    best_lr, searches, rows, results = {}, [], [], {}
    for opt in ("GD", "SGD", "MBGD", "AdaGrad", "Adam"):
        lr, df, _ = lr_search(d, opt); best_lr[opt] = lr; searches.append(df)
        r = fit_and_evaluate(d, opt, lr); results[opt] = r
        rows.append(summary_row(opt, r))
        history_frame(r["history"]).to_csv(os.path.join(res_dir, "training_histories", f"{opt}.csv"), index=False)
        build_model(d, r).save(os.path.join(root, "models", MODEL_FILES[opt]),
                               optimizer=opt, learning_rate=lr, basis_idx=d["basis_idx"])
    pd.concat(searches).to_csv(os.path.join(res_dir, "lr_search.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, "optimizer_comparison.csv"), index=False)
    kernel_comparison(d).to_csv(os.path.join(res_dir, "kernel_comparison.csv"), index=False)
    lin = linear_svm_reference(d); svc = sklearn_svc_baseline(d)
    approximation_check(d).to_csv(os.path.join(res_dir, "approximation_quality.csv"), index=False)
    exact_kernel_demo(d).to_csv(os.path.join(res_dir, "exact_kernel_demo.csv"), index=False)
    print(pd.DataFrame(rows).to_string())
    json.dump({"config": CONFIG, "gamma0": d["gamma0"], "gamma": d["gamma"], "selected_lr": best_lr,
               "linear_svm": {"lam": lin["lam"], "lr": lin["lr"]}, "svc_C": svc["C"]},
              open(os.path.join(res_dir, "hyperparameters.json"), "w"), indent=2)
    print("linear", lin["test"]["accuracy"], lin["test"]["macro_f1"], "svc", svc["test"]["accuracy"], svc["test"]["macro_f1"])


if __name__ == "__main__":
    main()
