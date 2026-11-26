"""Plotting helpers. Every experimental curve is drawn from values recorded during training."""
import os
import numpy as np
import matplotlib.pyplot as plt

COLORS = {"GD": "#1f77b4", "SGD": "#ff7f0e", "MBGD": "#2ca02c", "AdaGrad": "#9467bd", "Adam": "#d62728"}
NAMES = {"GD": "Gradient Descent", "SGD": "SGD", "MBGD": "Mini-Batch GD", "AdaGrad": "AdaGrad", "Adam": "Adam"}
SHORT_CLASSES = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]
FIG_DIR = "figures"

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def set_fig_dir(path):
    global FIG_DIR
    FIG_DIR = path
    os.makedirs(path, exist_ok=True)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.show()


def class_distribution(y_names_train, y_names_test, order, name="class_distribution.png"):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    tr = [int(np.sum(np.asarray(y_names_train) == c)) for c in order]
    te = [int(np.sum(np.asarray(y_names_test) == c)) for c in order]
    x = np.arange(len(order))
    ax.bar(x - 0.2, tr, 0.4, label="official train", color="#4c72b0")
    ax.bar(x + 0.2, te, 0.4, label="official test", color="#dd8452")
    for i, (a, b) in enumerate(zip(tr, te)):
        ax.text(i - 0.2, a + 15, str(a), ha="center", fontsize=8)
        ax.text(i + 0.2, b + 15, str(b), ha="center", fontsize=8)
    ax.set_xticks(x, order, rotation=20)
    ax.set_ylabel("windows"); ax.set_title("Class distribution (6 activities)"); ax.legend()
    save(fig, name)


def pca_projection(X, y, class_names, name="pca_projection.png"):
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    ev = s ** 2 / np.sum(s ** 2)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(class_names):
        m = y == k
        ax.scatter(P[m, 0], P[m, 1], s=5, alpha=0.5, color=cmap(k), label=c)
    ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}% var)"); ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}% var)")
    ax.set_title("PCA projection of the standardised training features\n(visualisation only - not the model's decision space)")
    ax.legend(markerscale=3, fontsize=8)
    save(fig, name)
    return ev


def feature_distributions(df, y_names, features, order, name="feature_distributions.png"):
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3.6))
    for ax, f in zip(np.atleast_1d(axes), features):
        data = [df.loc[np.asarray(y_names) == c, f].to_numpy() for c in order]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(order) + 1), SHORT_CLASSES)
        ax.set_title(f, fontsize=9)
    save(fig, name)


def correlation_heatmap(df, features, name="feature_correlation.png"):
    C = df[features].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), [f[:22] for f in features], rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)), [f[:22] for f in features], fontsize=7)
    fig.colorbar(im, ax=ax); ax.set_title("Correlation of selected features"); ax.grid(False)
    save(fig, name)


def gram_heatmap(G, y, class_names, title, name):
    order = np.argsort(y, kind="stable")
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(G[np.ix_(order, order)], cmap="viridis")
    ys = y[order]
    ticks = [np.flatnonzero(ys == k).mean() for k in range(len(class_names)) if np.any(ys == k)]
    ax.set_xticks(ticks, SHORT_CLASSES[:len(ticks)], fontsize=8); ax.set_yticks(ticks, SHORT_CLASSES[:len(ticks)], fontsize=8)
    ax.set_title(title); ax.grid(False); fig.colorbar(im, ax=ax)
    save(fig, name)


def rbf_similarity_curves(gamma0, multipliers, d2_max, name="rbf_similarity.png"):
    d2 = np.linspace(0, d2_max, 400)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in multipliers:
        ax.plot(d2, np.exp(-gamma0 * m * d2), label=f"gamma = {m:g} x median heuristic")
    ax.set_xlabel("squared distance ||x - z||^2 (standardised features)"); ax.set_ylabel("K(x, z)")
    ax.set_title("RBF similarity decays with distance - gamma controls how fast"); ax.legend(fontsize=8)
    save(fig, name)


def decision_boundaries(models, X, y, titles, name):
    """models: list of callables f(grid_points) -> class predictions (2-D synthetic data only)."""
    x0, x1 = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y0, y1 = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    grid = np.c_[gx.ravel(), gy.ravel()]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.4))
    for ax, f, t in zip(np.atleast_1d(axes), models, titles):
        Zp = f(grid).reshape(gx.shape)
        ax.contourf(gx, gy, Zp, alpha=0.25, levels=np.arange(-0.5, 3), cmap="brg")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=10, edgecolors="k", linewidths=0.2)
        ax.set_title(t, fontsize=10); ax.set_aspect("equal")
    fig.suptitle("SYNTHETIC 2-D demonstration (not the HAR results)", fontsize=10)
    save(fig, name)


def training_curves(histories, keys=("train_obj", "val_obj", "val_acc", "val_f1"), name="training_curves.png",
                    titles=None):
    titles = titles or {"train_obj": "Training objective J", "val_obj": "Validation objective",
                        "val_acc": "Validation accuracy", "val_f1": "Validation Macro F1",
                        "train_acc": "Training accuracy"}
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.6))
    for ax, k in zip(np.atleast_1d(axes), keys):
        for opt, h in histories.items():
            ax.plot(h[k], color=COLORS.get(opt), label=NAMES.get(opt, opt), lw=1.6)
        ax.set_title(titles.get(k, k)); ax.set_xlabel("epoch")
        if "obj" in k:
            ax.set_yscale("log")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    save(fig, name)


def lr_search_plot(df, opt, name):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ok = df[~df.diverged]
    ax.semilogx(ok.lr, ok.final_val_macro_f1, "o-", color=COLORS.get(opt))
    for _, r in df[df.diverged].iterrows():
        ax.axvline(r.lr, color="grey", ls=":", lw=1)
    ax.set_xlabel("learning rate"); ax.set_ylabel("final validation Macro F1")
    ax.set_title(f"{NAMES.get(opt, opt)}: learning-rate search (100 epochs each)")
    save(fig, name)


def confusion(cm, class_names, title, name, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(class_names)), SHORT_CLASSES[:len(class_names)], fontsize=8)
    ax.set_yticks(range(len(class_names)), SHORT_CLASSES[:len(class_names)], fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10); ax.grid(False)
    if own:
        save(fig, name)


def confusion_grid(cms, class_names, name="confusion_matrices_all.png"):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 4))
    for ax, (opt, cm) in zip(np.atleast_1d(axes), cms.items()):
        confusion(cm, class_names, NAMES.get(opt, opt), None, ax=ax)
    save(fig, name)


def bar_compare(values, ylabel, title, name, fmt="{:.3f}", log=False):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    keys = list(values)
    bars = ax.bar([NAMES.get(k, k) for k in keys], [values[k] for k in keys], color=[COLORS.get(k, "#777") for k in keys])
    for bar, k in zip(bars, keys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(values[k]), ha="center", va="bottom", fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel); ax.set_title(title)
    save(fig, name)


def per_class_f1(f1_by_opt, class_names, name="per_class_f1.png"):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(class_names)); w = 0.8 / len(f1_by_opt)
    for i, (opt, f) in enumerate(f1_by_opt.items()):
        ax.bar(x + (i - len(f1_by_opt) / 2 + 0.5) * w, f, w, color=COLORS.get(opt), label=NAMES.get(opt, opt))
    ax.set_xticks(x, class_names, rotation=15, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("test F1"); ax.set_title("Per-class F1 on the independent test set"); ax.legend(fontsize=7, ncol=5)
    save(fig, name)


def sensitivity_lines(df, x, y, group, title, name, logx=True, xlabel=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for g, sub in df.groupby(group):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], "o-", label=f"{group} = {g:g}")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or x); ax.set_ylabel(y); ax.set_title(title); ax.legend(fontsize=7)
    save(fig, name)
