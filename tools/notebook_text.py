"""Markdown text shared by the five notebooks (English). Optimizer-specific parts are keyed by optimizer."""

TITLE = "Human Activity Recognition Using Kernel-Based Multi-Class Support Vector Machines and Five Optimization Algorithms"

OPT_FULL = {"GD": "Gradient Descent (GD)", "SGD": "Stochastic Gradient Descent (SGD)",
            "MBGD": "Mini-Batch Gradient Descent (MBGD)", "AdaGrad": "AdaGrad", "Adam": "Adam"}

ABSTRACT = """## 2. Abstract
This notebook is one of five companion notebooks of Assignment 4. All five train the **same** model: a
kernelized, direct multi-class Support Vector Machine (Crammer-Singer hinge loss with RKHS regularisation)
written from scratch in NumPy, with the **RBF kernel** as the primary kernel. The task is to recognise six
human activities from smartphone accelerometer and gyroscope features (UCI HAR dataset, 561 features,
30 volunteers). The notebooks differ only in the optimizer that minimises the objective:
GD, SGD, Mini-Batch GD, AdaGrad and Adam. Data, preprocessing, kernel, kernel basis, initial parameters,
regularisation, objective, epoch budget and metrics are identical, so the comparison is controlled.
**This notebook focuses on {opt}.**"""

PROBLEM = """## 3. Problem Statement
Smartphones carry an accelerometer and a gyroscope. From a 2.56-second window of these signals we want to
decide **what the person is doing**: walking, walking upstairs, walking downstairs, sitting, standing or
lying down. This is a real-world problem (elderly monitoring, fitness tracking, health applications) and a
genuine **multi-class** problem with **K = 6** classes. Some classes are easy to separate (moving vs. static),
while others overlap strongly (sitting vs. standing), so a non-linear decision boundary may help."""

OBJECTIVES = """## 4. Project Objectives
1. Implement RBF, polynomial, linear (and exploratory sigmoid) kernels manually.
2. Implement a genuine kernelized direct multi-class SVM via the Representer Theorem.
3. Implement five optimizers manually (GD, SGD, Mini-Batch GD, AdaGrad, Adam) on the same objective.
4. Prevent data leakage (subject-aware validation, statistics fitted on training data only).
5. Select the kernel configuration on validation data, then compare optimizers fairly.
6. Evaluate on the untouched official test subjects with Macro F1 as the primary metric.
7. Compare against a Linear SVM and a scikit-learn `SVC` reference baseline."""

DATASET = """## 5. Dataset Description
**UCI Human Activity Recognition Using Smartphones** (Anguita et al., ESANN 2013).

| Item | Value |
|---|---|
| Source (UCI) | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| Kaggle mirror | https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones |
| Licence | CC BY 4.0 (UCI listing); the original README asks to cite [1] and prohibits commercial use |
| Volunteers | 30 (age 19-48), Samsung Galaxy S II on the waist |
| Signals | 3-axial accelerometer + gyroscope, 50 Hz, windows of 2.56 s with 50% overlap |
| Features | 561 time- and frequency-domain features, already scaled to [-1, 1] by the authors |
| Official split | 7,352 training windows (21 subjects) / 2,947 test windows (9 different subjects) |
| Classes | WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING |

The loader below accepts both the original UCI text files and the Kaggle CSV version."""

THEORY_SVM = r"""## 10. SVM Theory
**Hyperplane and margin.** A linear classifier uses a hyperplane $w^\top x + b = 0$. Many hyperplanes may
separate two classes; the **Support Vector Machine** chooses the one with the **maximum margin**, i.e. the
largest distance $2/\lVert w\rVert$ between the two parallel hyperplanes $w^\top x + b = \pm 1$.
The training points lying on (or inside) the margin are the **support vectors**: only they determine the
solution.

* **Hard margin:** $\min \tfrac12\lVert w\rVert^2$ s.t. $y_i(w^\top x_i+b)\ge 1$ - requires perfectly separable data.
* **Soft margin:** allows violations through the hinge loss $\max(0, 1 - y_i f(x_i))$:
  $\min_w \tfrac{\lambda}{2}\lVert w\rVert^2 + \tfrac1N\sum_i \max(0, 1-y_i f(x_i))$. The **regularisation**
  coefficient $\lambda$ trades margin width against training errors (it plays the role of $1/(CN)$ in the
  classical $C$-formulation).

**Linear vs. non-linear classification.** If the classes are arranged, for example, as concentric rings,
no straight line (hyperplane) in the original space can separate them. Mapping the data to a richer feature
space $\phi(x)$ can make them linearly separable there; the separating hyperplane in feature space
corresponds to a **curved** boundary in the original space."""

THEORY_KERNEL = r"""## 11. Kernel Trick Explanation
A **kernel** is a function $k(x,z) = \langle \phi(x), \phi(z)\rangle$ that returns the inner product of two
samples in a (possibly infinite-dimensional) feature space **without constructing $\phi$ explicitly**.
For the RBF kernel the feature space is infinite-dimensional, yet $k(x,z)=\exp(-\gamma\lVert x-z\rVert^2)$
costs only one distance computation - this is the **kernel trick**.

* **Kernel (Gram) matrix:** $G_{ij} = k(z_i, z_j)$ collects all pairwise similarities.
* **Positive semidefinite (PSD):** a valid (Mercer) kernel produces $G \succeq 0$ for every finite set of
  points, i.e. $a^\top G a \ge 0$. PSD guarantees that $k$ is a real inner product, that
  $\lVert f\rVert^2 = a^\top G a \ge 0$, and that the regularised objective is convex.
* **RKHS:** every PSD kernel defines a *Reproducing Kernel Hilbert Space* $\mathcal H$ of functions
  $f(x)=\sum_j a_j k(x, z_j)$ with norm $\lVert f\rVert_{\mathcal H}^2 = a^\top G a$ and the reproducing
  property $f(x) = \langle f, k(\cdot, x)\rangle_{\mathcal H}$.
* **Representer Theorem:** the minimiser of (loss on the training data) + $\lambda\lVert f\rVert^2_{\mathcal H}$
  can be written as a weighted sum of kernels centred at the training points. We therefore optimise the
  coefficient matrix $A$ instead of an (infinite) weight vector.

**Kernels implemented** (Section 15):

| Kernel | Formula | Notes |
|---|---|---|
| Linear | $x^\top z$ | identical to the Linear SVM of the previous assignment |
| Polynomial | $(\gamma x^\top z + r)^d$ | feature interactions up to degree $d$; large $d$ -> overfitting / numerical growth |
| RBF | $\exp(-\gamma\lVert x-z\rVert^2)$ | local similarity in (0, 1], $k(x,x)=1$; **primary kernel** |
| Sigmoid | $\tanh(\gamma x^\top z + r)$ | inspired by neural activations; **not PSD in general** -> exploratory only |

**Effect of RBF $\gamma$.** Small $\gamma$: every sample influences far-away points, boundaries are smooth
(close to linear, risk of underfitting). Large $\gamma$: each sample influences only its close neighbourhood,
boundaries become very wiggly (risk of overfitting; $G \to I$). We initialise $\gamma$ with the **median
heuristic** $\gamma_0 = 1/\mathrm{median}\lVert x_i-x_j\rVert^2$ and tune a multiplier on validation data.

**Feature scaling matters for RBF:** the kernel depends on Euclidean distances, so a feature with a larger
numeric range would dominate the distance. Standardisation puts all 561 features on the same scale."""

THEORY_MULTI = r"""## 12. Multi-Class Classification Theory
Binary SVMs separate two classes. Common multi-class strategies:

* **One-vs-Rest (OvR):** K binary SVMs, class k against all others; predict the largest score.
* **One-vs-One (OvO):** K(K-1)/2 = 15 binary SVMs, one per pair; predict by voting
  (scikit-learn `SVC` uses OvO internally).
* **Direct multi-class (Crammer-Singer):** a **single** optimisation problem with one score function per
  class, trained jointly. The loss asks the correct class to beat the **best wrong class** by a margin of 1.

We use the direct Crammer-Singer formulation: one objective, all classes coupled, no voting."""

THEORY_OBJECTIVE = r"""## 13. Mathematical Objective
Let $Z = \{z_1,\dots,z_M\}$ be the kernel **basis** (training samples), $H_{ij} = k(x_i, z_j)$ the
$N\times M$ kernel feature matrix, $G_{ij} = k(z_i, z_j)$ the $M\times M$ basis Gram matrix,
$A\in\mathbb R^{M\times K}$ the coefficients and $b\in\mathbb R^K$ the biases.

**Scores:** $S = HA + b$, i.e. $f_k(x) = \sum_j A_{jk}\,k(x, z_j) + b_k$; **prediction** $\hat y = \arg\max_k S_{ik}$.

**Objective (regularised Crammer-Singer kernel hinge loss):**
$$J(A,b) = \frac{\lambda}{2}\,\mathrm{Tr}(A^\top G A) + \frac1N\sum_{i=1}^N L_i,\qquad
L_i = \max\Big(0,\; 1 + \max_{j\ne y_i} S_{ij} - S_{i,y_i}\Big).$$
$\mathrm{Tr}(A^\top G A) = \sum_k \lVert f_k\rVert^2_{\mathcal H}$ is the **RKHS norm** (not an ordinary
$\lVert A\rVert^2$ penalty). The bias is not regularised.

**Subgradient.** Define $D\in\mathbb R^{N\times K}$: if sample $i$ violates the margin ($L_i>0$), put $-1$ at
the true class $y_i$ and $+1$ at the highest-scoring wrong class $j^*$; otherwise the row is zero
(ties: the smallest class index among the maximal wrong classes).
$$\frac{\partial J}{\partial A} = \frac{H^\top D}{N} + \lambda G A,\qquad \frac{\partial J}{\partial b} = \frac{1}{N}\sum_i D_{i,:}.$$
For SGD / mini-batches the data term is averaged over the batch $B$ ($H_B^\top D_B/|B|$) while the
regularisation gradient $\lambda G A$ is kept exactly - this is an unbiased estimate of the full subgradient.

**Why the boundary is non-linear:** $f_k(x)$ is a weighted sum of RBF bumps centred at the basis samples, so
$f_k$ is a non-linear function of $x$ and the decision regions $\{x: f_k(x) > f_j(x)\ \forall j\}$ are curved
in the original 561-dimensional space.

**What this is - and what it is not.**

| Formulation | Optimised variables | Used here? |
|---|---|---|
| Classical binary SVM **dual** (QP with box constraints, solved by SMO in libsvm/`SVC`) | dual variables $\alpha_i$ | only inside the `SVC` reference baseline |
| **Direct multi-class kernel SVM in coefficient space** (this project) | $A, b$ by (sub)gradient methods | **yes - primary model** |
| Explicit feature approximation (Nystroem / Random Fourier Features) | linear weights $W$ on $\phi(x)$ | only in a verification experiment |

Our problem is an *unconstrained regularised risk minimisation* in RKHS coefficient space; it is **not** the
conventional constrained SVM dual."""

OPT_THEORY = {
    "GD": r"""## 14. Optimizer Theory - Gradient Descent (GD)
Full-batch GD uses the subgradient on **all** $N$ training samples and makes **one update per epoch**:
$$\theta_{t+1} = \theta_t - \eta\, g_t,\qquad g_t = \nabla J(\theta_t).$$
* **Advantages:** exact (non-noisy) subgradient, deterministic, monotone behaviour for small $\eta$, easy to analyse.
* **Limitations for kernel methods:** each update costs $O(NMK)$ for $H^\top D$ plus $O(M^2K)$ for $GA$; only one
  update per pass over the data; the coefficient-space problem is **ill-conditioned** because RBF kernel columns
  are strongly correlated (the largest eigenvalue of $H^\top H/N$ is orders of magnitude larger than the median one), so
  progress along most directions is slow, while too large steps make the non-smooth subgradient iterates oscillate.""",
    "SGD": r"""## 14. Optimizer Theory - Stochastic Gradient Descent (SGD)
SGD processes **exactly one sample per update** ($N$ updates per epoch) in a reproducibly shuffled order:
$$\theta_{t+1} = \theta_t - \eta\,\big(\nabla L_{i_t}(\theta_t) + \nabla R(\theta_t)\big),\qquad \nabla_A R = \lambda G A .$$
The single-sample subgradient is an unbiased but **noisy** estimate of the full subgradient, so the objective
fluctuates, but many cheap updates are made per epoch.

**Efficient exact implementation.** The naive update needs $\lambda GA$ ($O(M^2K)$) for every one of the $N$
updates. With the eigendecomposition $G = U\,\mathrm{diag}(\Lambda)\,U^\top$ and rotated coefficients $C = U^\top A$,
the identical update becomes $C \leftarrow (1-\eta\lambda\Lambda)\odot C - \eta\,(h_iU)^\top d_i$, costing $O(MK)$.
Because $U$ is orthogonal this produces **exactly the same iterates** as the naive SGD (verified numerically in
this notebook and in `tests/test_optimizers.py`). The one-off cost of the eigendecomposition is included in the
SGD training time.""",
    "MBGD": r"""## 14. Optimizer Theory - Mini-Batch Gradient Descent (MBGD)
MBGD averages the subgradient over a mini-batch $B$ (here $|B|=64$; the last batch of an epoch is smaller):
$$\theta_{t+1} = \theta_t - \eta\Big(\frac{1}{|B|}\sum_{i\in B}\nabla L_i(\theta_t) + \nabla R(\theta_t)\Big).$$
It is the compromise between GD (exact, 1 update/epoch) and SGD (noisy, $N$ updates/epoch): with $N=5{,}551$ and
$|B|=64$ there are $\lceil 5551/64\rceil = 87$ updates per epoch. For kernel SVMs mini-batches are attractive
because $H_B^\top D_B$ is a dense matrix product that NumPy/BLAS executes efficiently, and the noise is reduced by
a factor of about $\sqrt{|B|}$ compared with SGD.""",
    "AdaGrad": r"""## 14. Optimizer Theory - AdaGrad
AdaGrad gives **every parameter its own learning rate**, shrinking it for parameters that have received large
gradients:
$$\text{acc}_t = \text{acc}_{t-1} + g_t^2,\qquad \theta_{t+1} = \theta_t - \frac{\eta\, g_t}{\sqrt{\text{acc}_t}+\epsilon}.$$
(The accumulator is often written $G_t$; in the code it is called `sq_grad_sum` so that it cannot be confused with
the kernel Gram matrix $G$.) Separate accumulators are kept for $A$ and $b$, initialised to zero, $\epsilon=10^{-8}$,
mini-batches of 64. Because the accumulator only grows, the effective step size $\eta/(\sqrt{\text{acc}}+\epsilon)$
decreases monotonically - helpful against noise but it can stop learning too early.""",
    "Adam": r"""## 14. Optimizer Theory - Adam
Adam combines **momentum** (first moment) with **RMSProp-style scaling** (second moment) and corrects the bias of
the zero-initialised moments:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t},\qquad
\theta_{t+1} = \theta_t - \frac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$
Defaults: $\eta = 10^{-3}$ (tuned), $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, mini-batches of 64, global update
counter $t$. Because the step is normalised per parameter, Adam is largely insensitive to the poor conditioning of
the kernel coefficient problem - but it is **not guaranteed** to be the best optimizer.""",
}

PREPROC = r"""## 9. Preprocessing
1. **Feature / target separation** - the 561 numeric features form $X$; `Activity` is the target; the
   `subject` id is kept **only** for splitting and is **never** used as a feature.
2. **Label encoding** - fixed official order WALKING=0, ..., LAYING=5.
3. **Subject-aware validation split** - the official training set (21 subjects) is split with
   `GroupShuffleSplit(test_size=0.2, random_state=42)` so that validation subjects are unseen, like the official
   test subjects. The **official test set is used only once, for the final evaluation**.
4. **Standardisation** - mean and standard deviation are computed on the training split only and applied to
   validation and test.
5. **Kernel basis** - selected **only from the training split** (class-stratified, seed 42).
6. **RBF $\gamma$** - median heuristic computed on the training split only."""

REFERENCES = """## 24. References
1. D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013.
2. C. Cortes, V. Vapnik. *Support-Vector Networks.* Machine Learning 20, 1995.
3. K. Crammer, Y. Singer. *On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines.* JMLR 2, 2001.
4. B. Schoelkopf, R. Herbrich, A. Smola. *A Generalized Representer Theorem.* COLT 2001.
5. B. Schoelkopf, A. Smola. *Learning with Kernels.* MIT Press, 2002.
6. C. Williams, M. Seeger. *Using the Nystroem Method to Speed Up Kernel Machines.* NIPS 2001.
7. A. Rahimi, B. Recht. *Random Features for Large-Scale Kernel Machines.* NIPS 2007.
8. J. Duchi, E. Hazan, Y. Singer. *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.* JMLR 2011.
9. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
10. C.-C. Chang, C.-J. Lin. *LIBSVM: A Library for Support Vector Machines.* ACM TIST 2011 (used by scikit-learn `SVC`)."""
