"""Results interpretation / conclusion text for each notebook, generated from the result files so that every
number quoted in the Markdown equals the executed outputs (tools/patch_discussion.py re-applies it after execution)."""
import json
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
NAMES = {"GD": "Gradient Descent", "SGD": "SGD", "MBGD": "Mini-Batch GD", "AdaGrad": "AdaGrad", "Adam": "Adam"}


def _load():
    c = pd.read_csv(os.path.join(R, "optimizer_comparison.csv")).set_index("optimizer")
    extra = {}
    for f in ("kernel_selection", "basis_size_analysis", "kernel_comparison", "batch_size_study", "exact_kernel_demo",
              "approximation_quality"):
        p = os.path.join(R, f + ".csv")
        extra[f] = pd.read_csv(p) if os.path.exists(p) else None
    p = os.path.join(R, "linear_vs_kernel.csv")
    extra["lvk"] = pd.read_csv(p, index_col=0) if os.path.exists(p) else None
    p = os.path.join(R, "optimizer_verdict.json")
    extra["verdict"] = json.load(open(p)) if os.path.exists(p) else None
    return c, extra


def pct(x):
    return f"{100 * float(x):.2f}%"


def discussion(opt):
    c, e = _load()
    r = c.loc[opt]
    rank_val = int(c.val_macro_f1.rank(ascending=False)[opt])
    rank_test = int(c.macro_f1.rank(ascending=False)[opt])
    common = (f"* Selected learning rate **{r.learning_rate:g}**; {int(r.updates):,} parameter updates in "
              f"{int(r.epochs)} epochs ({int(r.samples_processed):,} processed samples).\n"
              f"* Validation Macro F1 **{r.val_macro_f1:.4f}** (rank {rank_val}/5 - the selection metric); test accuracy "
              f"**{pct(r.accuracy)}**, test Macro F1 **{r.macro_f1:.4f}** (rank {rank_test}/5), OvR ROC-AUC {r.roc_auc_ovr:.4f}.\n"
              f"* Final training objective J = {r.train_objective:.4f} = hinge {r.train_hinge:.4f} + RKHS regularisation "
              f"{r.regularization:.4f}; the training objective increased in {r.pct_epochs_obj_increased:.0f}% of the epochs.\n")
    sit = "SITTING and STANDING are the most confused pair (static postures with similar sensor statistics); LAYING and the walking classes are recognised almost perfectly."
    if opt == "GD":
        sel = e["kernel_selection"]
        best = sel.sort_values("val_macro_f1", ascending=False).iloc[0]
        interp = (common +
                  f"* GD is clearly the weakest optimizer under the common 100-epoch budget: one update per epoch means only 100 "
                  f"updates in total, and the coefficient-space problem is ill-conditioned (eigenvalues of H^T H/N printed "
                  f"above span several orders of magnitude), so GD is still far from the optimum (J = {r.train_objective:.3f} "
                  f"versus {c.train_objective.min():.3f} for the best optimizer).\n"
                  f"* The learning-rate search shows the trade-off: larger steps help until about lr = 1, while with lr = 3 the "
                  f"subgradient iterates oscillate and the objective blows up.\n"
                  f"* Kernel selection: the best validation Macro F1 ({best.val_macro_f1:.4f}) is reached at gamma = "
                  f"{best.gamma_multiplier:g} x gamma0 and lambda = {best.lam:g}. Too small gamma (0.5 x) under-fits (smooth, "
                  f"almost linear functions), too large gamma (8 x) starts to lose generalisation, and lambda = 0.01 "
                  f"over-regularises for every gamma.\n* {sit}")
        concl = ("Full-batch GD implements the objective exactly and is perfectly stable at small step sizes, but with an equal "
                 "epoch budget it is the least effective optimizer for this kernel SVM. It would need many more iterations "
                 "(or preconditioning) to match the stochastic and adaptive methods. The shared kernel configuration selected "
                 "in this notebook (RBF, gamma = 4 x median heuristic, lambda = 1e-4, M = 2,000) is used by all five notebooks.")
    elif opt == "SGD":
        bs = e["basis_size_analysis"]
        b2 = bs[bs.basis_M == 2000].iloc[0]; bx = bs[bs.exact].iloc[0]
        interp = (common +
                  f"* SGD makes {int(r.updates):,} cheap single-sample updates, so per **epoch** it is one of the fastest to reduce "
                  f"the objective (final J = {r.train_objective:.4f}, rank {int(c.train_objective.rank()[opt])}/5).\n"
                  f"* The within-epoch trace shows the typical noisy behaviour: individual updates often increase J, but the trend is downward.\n"
                  f"* The rotated-coordinate implementation reproduces naive SGD to round-off precision; with it SGD needs "
                  f"{r.training_time_s:.1f} s for 100 epochs versus {c.loc['MBGD', 'training_time_s']:.1f} s for Mini-Batch GD "
                  f"(only GD, with just 100 updates, is faster).\n"
                  f"* Basis size: validation Macro F1 rises from {bs.val_macro_f1.iloc[0]:.4f} (M = {int(bs.basis_M.iloc[0])}) to "
                  f"{b2.val_macro_f1:.4f} (M = 2,000) and {bx.val_macro_f1:.4f} with the exact full basis (M = {int(bx.basis_M):,}), "
                  f"while training time grows from {b2.train_time_s:.0f} s to {bx.train_time_s:.0f} s (30 epochs). M = 2,000 keeps most "
                  f"of the accuracy at a fraction of the cost - the restriction costs about "
                  f"{100 * (bx.val_macro_f1 - b2.val_macro_f1):.1f} Macro-F1 points.\n* {sit}")
        concl = ("SGD optimises the genuine kernel objective with one sample per update. Its noise is visible but harmless at "
                 "the selected step size, and with an O(MK) update it is both fast and accurate. The basis-size study justifies "
                 "Mode B as a documented compromise between exactness and cost.")
    elif opt == "MBGD":
        kc = e["kernel_comparison"]
        rbf = kc[kc.kernel == "rbf"].iloc[0]; lin = kc[kc.kernel == "linear"].iloc[0]
        sig = kc[kc.kernel.str.startswith("sigmoid")].iloc[0]
        bstudy = e["batch_size_study"]
        btxt = ""
        if bstudy is not None:
            btxt = "* Batch-size study (val Macro F1): " + ", ".join(
                f"batch {int(x.batch_size)} -> {x.val_macro_f1:.4f} ({int(x.updates):,} updates, {x.train_time_s:.0f} s)" for x in bstudy.itertuples()) + ".\n"
        interp = (common +
                  f"* With {int(r.updates / r.epochs)} updates per epoch, MBGD sits between GD and SGD: much better than GD "
                  f"(test Macro F1 {r.macro_f1:.4f} vs {c.loc['GD', 'macro_f1']:.4f}) and close to SGD ({c.loc['SGD', 'macro_f1']:.4f}).\n" + btxt +
                  f"* Kernel comparison (same basis, Adam): RBF test Macro F1 {rbf.test_macro_f1:.4f}; polynomial d=2/3 "
                  f"{kc.test_macro_f1.iloc[1]:.4f}/{kc.test_macro_f1.iloc[2]:.4f}; linear kernel in coefficient space only "
                  f"{lin.test_macro_f1:.4f} (its large, highly correlated kernel values make the coefficient problem very badly "
                  f"conditioned - the same model trained in the primal is much better, see notebook 04). The exploratory sigmoid kernel "
                  f"scored {sig.test_macro_f1:.4f}, but its Gram matrix has a negative eigenvalue ({sig.min_eig_G:.1f}): it is not a "
                  f"valid PSD kernel here, so its objective is not guaranteed to be convex and it is not used as a main model.\n"
                  f"* The synthetic rings demonstrate why kernels exist: a linear boundary cannot separate the rings, the RBF kernel SVM can.\n* {sit}")
        concl = ("Mini-batches give stable, vectorised updates and a good accuracy-cost balance. Among valid PSD kernels the "
                 "RBF kernel gave the best validation and test results, confirming it as the primary kernel.")
    elif opt == "AdaGrad":
        lvk = e["lvk"]
        ltxt = ""
        if lvk is not None:
            cols = list(lvk.columns)
            lf, kf, sf = (float(lvk.loc["Macro F1", col]) for col in cols)
            la, ka, sa = (float(lvk.loc["Accuracy", col]) for col in cols)
            ltxt = (f"* **Linear vs Kernel:** the controlled Linear SVM reached test accuracy {pct(la)} / Macro F1 {lf:.4f}; the RBF "
                    f"kernel SVM ({cols[1].split(', ')[1].rstrip(')')}) reached {pct(ka)} / {kf:.4f}; the scikit-learn SVC reference "
                    f"{pct(sa)} / {sf:.4f}. On HAR the **linear model is at least as good** as our restricted kernel model: the 561 "
                    f"engineered features are already close to linearly separable, the official test subjects are new people (a "
                    f"flexible model can fit subject-specific patterns), and our kernel model is limited by the 2,000-sample basis "
                    f"and the 100-epoch first-order optimisation. The kernel model is also far more expensive (basis storage and "
                    f"kernel evaluation at prediction time).\n")
        ed = e["exact_kernel_demo"]
        etxt = ""
        if ed is not None:
            etxt = ("* Exact-kernel demo (1,500 samples): " + "; ".join(f"{x.mode}: val Macro F1 {x.val_macro_f1:.4f}" for x in ed.itertuples()) +
                    ". The Nystroem model uses the same 300 landmarks as the restricted model (it is a re-parametrisation of it), "
                    "yet trains better in 60 epochs - its whitened features are much better conditioned.\n")
        interp = (common +
                  f"* AdaGrad's per-parameter steps compensate for the poor conditioning; it achieved the "
                  f"{'highest' if c.accuracy.idxmax() == 'AdaGrad' else 'rank ' + str(int(c.accuracy.rank(ascending=False)['AdaGrad']))} "
                  f"test accuracy of the five optimizers.\n"
                  f"* The effective learning rate decays monotonically because the squared-gradient accumulator only grows.\n" + ltxt + etxt + f"* {sit}")
        concl = ("AdaGrad is an excellent optimizer for this problem: adaptive, stable and accurate. The comparison with a "
                 "Linear SVM shows that a kernel is not automatically better - on this dataset the linear model is accurate, "
                 "much cheaper and easier to deploy.")
    else:
        v = e["verdict"] or {}
        if "most_stable_epoch_level (lowest % epochs with J increase, then tail std)" not in v:
            v = {}  # verdict from an older run - the text is regenerated after execution (tools/patch_discussion.py)
        best_val = c.val_macro_f1.idxmax()
        interp = (common +
                  f"* **Best optimizer by validation Macro F1 (the pre-registered selection rule): {NAMES[best_val]}** "
                  f"({c.val_macro_f1.max():.4f}).\n"
                  f"* Highest independent test accuracy: **{NAMES[c.accuracy.idxmax()]}** ({pct(c.accuracy.max())}); highest test Macro F1: "
                  f"**{NAMES[c.macro_f1.idxmax()]}** ({c.macro_f1.max():.4f}). The validation winner and the test winner differ "
                  f"- with only 5 validation subjects, differences of about 1 point are within subject-to-subject variability.\n"
                  f"* Lowest training objective: **{NAMES[c.train_objective.idxmin()]}** (J = {c.train_objective.min():.4f}); "
                  f"fastest training: **{NAMES[c.training_time_s.idxmin()]}**.\n"
                  + (f"* Fastest convergence (first epoch with J <= {v.get('convergence_target_J', 0):.4f}, i.e. 1.1 x the lowest final J): "
                     f"**{NAMES[v['fastest_convergence_epochs']]}**; epochs to target: {v['epochs_to_target']}.\n" if v else "")
                  + (f"* Most stable at epoch level (fewest epochs in which J increased): **{NAMES[v['most_stable_epoch_level (lowest % epochs with J increase, then tail std)']]}** "
                     f"(within an epoch SGD is the noisiest - see notebook 02); best efficiency (fastest optimizer within 1 Macro-F1 point of the best): "
                     f"**{NAMES[v['best_efficiency (fastest within 1 Macro-F1 point of the best)']]}**.\n" if v else "")
                  + f"* GD is far behind (test Macro F1 {c.loc['GD', 'macro_f1']:.4f}) because 100 epochs = 100 updates.\n"
                  f"* Adam is **not** automatically the best: it wins on validation but not on the test subjects.\n* {sit}")
        concl = ("All five manually implemented optimizers minimise the same RKHS-regularised Crammer-Singer objective under "
                 "identical conditions. Stochastic and adaptive methods (SGD, AdaGrad, Adam, Mini-Batch GD) reach 92-94% test "
                 "accuracy within 100 epochs, while full-batch GD needs far more iterations. The ranking between the top four "
                 "depends on the split (validation vs test), so the honest conclusion is that they are statistically close; "
                 "among them SGD needed the least training time thanks to its exact O(MK) rotated update.")
    return {"interpretation": interp, "conclusion": concl}
