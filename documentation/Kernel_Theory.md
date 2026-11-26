# Kernel Theory for Support Vector Machines

## 1. Support Vector Machines
* **Hyperplane:** $\{x : w^\top x + b = 0\}$ - a flat decision surface (a line in 2-D, a plane in 3-D).
* **Margin:** distance between the hyperplane and the closest training points; for the canonical scaling
  $y_i(w^\top x_i+b)\ge1$ the margin width is $2/\lVert w\rVert$.
* **Maximum margin:** among all separating hyperplanes, choose the one with the widest margin - it is the most robust
  to small perturbations and has good generalisation guarantees.
* **Support vectors:** training points on or inside the margin; removing any other point does not change the solution.
* **Hard margin:** no violations allowed - only possible for separable data, very sensitive to outliers.
* **Soft margin:** violations are allowed but penalised by the **hinge loss** $\max(0, 1 - y f(x))$; the regularisation
  coefficient ($\lambda$, or $C\approx 1/(\lambda N)$) controls the trade-off between margin width and training errors.

## 2. Linear vs non-linear classification
A linear classifier can only draw straight boundaries. Data such as concentric rings, XOR patterns, or - in HAR -
the curved, overlapping clouds of SITTING and STANDING are not separable by a hyperplane in the original space.
Mapping $x \mapsto \phi(x)$ into a higher-dimensional **feature space** (e.g. adding $x_1^2, x_2^2, x_1x_2$) can make
them linearly separable there; the hyperplane in feature space is a curved boundary in the input space.

## 3. The kernel trick
The SVM solution and predictions depend on the data only through inner products $\phi(x)^\top\phi(z)$.
A **kernel** computes this inner product directly: $k(x,z)=\phi(x)^\top\phi(z)$. Example: for
$k(x,z)=(x^\top z)^2$ in 2-D, $\phi(x)=(x_1^2,\sqrt2x_1x_2,x_2^2)$ - the kernel needs 2 multiplications instead of building
$\phi$. For RBF, $\phi$ is infinite-dimensional, so the trick is the **only** practical way.

* **Kernel / Gram matrix:** $G_{ij}=k(z_i,z_j)$.
* **Positive semidefinite:** $a^\top G a\ge0$ for all $a$ and all point sets (Mercer's condition). Needed so that $k$ is
  a genuine inner product, the RKHS norm is non-negative, and the optimisation problem is convex.
* **RKHS:** the Hilbert space of functions $f=\sum_j a_jk(\cdot,z_j)$ (and limits) with
  $\langle k(\cdot,x),k(\cdot,z)\rangle=k(x,z)$ (reproducing property) and $\lVert f\rVert^2=a^\top Ga$. Small RKHS norm = smooth function.

## 4. Kernels used

### Linear $k(x,z)=x^\top z$
Feature space = input space; with the RKHS norm it is exactly the Linear SVM ($\lVert f\rVert^2=\lVert w\rVert^2$ with
$w=Z^\top a$). Used as the controlled Linear SVM reference.

### Polynomial $k(x,z)=(\gamma x^\top z+r)^d$
Contains all monomials (feature interactions) up to degree $d$. $\gamma$ scales the inner product, $r\ge0$ weights
lower-order terms, $d$ sets the complexity. PSD for integer $d\ge1$, $\gamma>0$, $r\ge0$. Large $d$ increases overfitting
risk and numerical range ($|\gamma x^\top z + r|^d$ can explode or vanish).

### RBF (Gaussian) $k(x,z)=\exp(-\gamma\lVert x-z\rVert^2)$  - primary kernel
* Similarity is 1 for identical points and decays with Euclidean distance: a **local** influence.
* $\gamma$ = inverse squared length-scale. **Small $\gamma$**: wide bumps, smooth boundaries, near-linear behaviour,
  underfitting risk. **Large $\gamma$**: narrow bumps, every training point becomes its own island,
  $G\to I$, overfitting and poor generalisation to new subjects.
* Always PSD (for $\gamma>0$); bounded in $(0,1]$, which makes it robust to outliers.
* Initialisation: **median heuristic** $\gamma_0=1/\mathrm{median}\lVert x_i-x_j\rVert^2$; we tuned $\gamma=c\cdot\gamma_0$ on validation.

### Sigmoid $k(x,z)=\tanh(\gamma x^\top z+r)$  - exploratory only
Mimics a neural-network activation (a two-layer network with tanh units). It is **not PSD in general** (we measure
negative eigenvalues of its Gram matrix), so the objective may be non-convex and the RKHS interpretation fails. It is
reported only as an exploratory comparison.

## 5. Why scaling matters for RBF
$\lVert x-z\rVert^2=\sum_f (x_f-z_f)^2$ adds the squared differences of all 561 features. A feature measured on a larger
scale would dominate the sum and the kernel would effectively ignore the others. Standardisation (fitted on the
training split only) gives every feature unit variance, and $\gamma$ becomes comparable across datasets.

## 6. Multi-class strategies
| Strategy | Models trained | Prediction | Used by |
|---|---|---|---|
| One-vs-Rest | $K$ | largest score | many linear SVM libraries |
| One-vs-One | $K(K-1)/2$ | majority vote | scikit-learn `SVC` (libsvm) |
| Direct (Crammer-Singer) | 1 joint problem | largest score | **this project** |

Crammer-Singer couples all classes in one loss: the true class must beat the strongest competitor by a margin of 1.

## 7. Scalability
An exact kernel machine stores an $N\times N$ Gram matrix ($O(N^2)$ memory, $O(N^2d)$ time to build) and predicts in
$O(N)$ kernel evaluations per sample. Remedies: a reduced basis (Mode B, used here), Nystroem low-rank approximation
or Random Fourier Features (Mode C). Linear SVMs need only $O(dK)$ parameters and $O(dK)$ prediction cost - the
reason they can be preferred on large or already linearly separable data.
