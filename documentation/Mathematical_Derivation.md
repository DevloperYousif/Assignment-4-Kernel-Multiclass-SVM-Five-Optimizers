# Mathematical Derivation - Kernelized Direct Multi-Class SVM

## 1. Notation

| Symbol | Shape | Meaning |
|---|---|---|
| $X$ | $N \times d$ | standardised training features ($N = 5{,}551$, $d = 561$) |
| $y_i \in \{0,\dots,K-1\}$ | | class label, $K = 6$ |
| $Z = \{z_1,\dots,z_M\}$ | $M \times d$ | kernel basis (training samples), $M = 2{,}000$ |
| $H_{ij} = k(x_i, z_j)$ | $N \times M$ | kernel feature matrix |
| $G_{ij} = k(z_i, z_j)$ | $M \times M$ | basis Gram matrix (symmetric, PSD) |
| $A$ | $M \times K$ | trainable coefficients |
| $b$ | $K$ | trainable biases |
| $\lambda$ | scalar | regularisation coefficient ($10^{-4}$) |

## 2. From the RKHS problem to coefficient space

We look for $K$ functions $f_1,\dots,f_K$ in the RKHS $\mathcal H$ of a PSD kernel $k$:
$$\min_{f_1..f_K,\,b}\ \frac{\lambda}{2}\sum_{k=1}^K \lVert f_k\rVert_{\mathcal H}^2 + \frac1N\sum_{i=1}^N L\big(y_i, f(x_i)+b\big).$$
**Representer Theorem.** Because the loss depends on the $f_k$ only through their values at the training points and
the regulariser is an increasing function of $\lVert f_k\rVert$, each minimiser has the form
$f_k(\cdot)=\sum_{j=1}^N A_{jk}\,k(\cdot, x_j)$. We restrict the expansion to $M$ basis samples $z_j$
(**Mode B**; with $M=N$ it is exact, **Mode A**):
$$f_k(x)=\sum_{j=1}^M A_{jk}\,k(x,z_j),\qquad \lVert f_k\rVert_{\mathcal H}^2=\sum_{j,l}A_{jk}A_{lk}k(z_j,z_l)=a_k^\top G a_k .$$
Hence $\sum_k\lVert f_k\rVert^2 = \mathrm{Tr}(A^\top G A)$ and, for all training samples at once, $S = HA + \mathbf 1 b^\top$.

## 3. Crammer-Singer multi-class hinge loss

$$L_i = \max\Big(0,\ 1 + \max_{j\neq y_i} S_{ij} - S_{i y_i}\Big)$$
The correct class must beat the **best wrong class** by at least 1. The inner maximum excludes $y_i$.

Final objective:
$$\boxed{J(A,b)=\frac{\lambda}{2}\mathrm{Tr}(A^\top G A)+\frac1N\sum_{i=1}^N L_i}$$
$J$ is **convex** in $(A,b)$: the hinge term is a maximum of affine functions of $(A,b)$, and $\mathrm{Tr}(A^\top G A)$ is a
convex quadratic because $G \succeq 0$.

## 4. Subgradients

Let $j_i^* = \arg\max_{j\ne y_i} S_{ij}$ (smallest index on ties) and $m_i = 1 + S_{i j_i^*} - S_{i y_i}$.

* If $m_i > 0$: $L_i = 1 + S_{i j^*} - S_{i y_i}$, so $\partial L_i/\partial S_{i j^*} = +1$, $\partial L_i/\partial S_{i y_i} = -1$, zero elsewhere.
* If $m_i < 0$: $L_i = 0$, gradient zero.
* If $m_i = 0$ (kink): any convex combination is a valid subgradient; we use **zero** (consistent rule).

Collect these in $D\in\mathbb R^{N\times K}$ ($D_{i,:}$ sums to zero). With $S = HA + \mathbf 1 b^\top$:
$$\frac{\partial}{\partial A}\frac1N\sum_i L_i = \frac{1}{N}H^\top D,\qquad \frac{\partial}{\partial b}\frac1N\sum_i L_i = \frac1N D^\top\mathbf 1 .$$
For the regulariser, $\frac{\partial}{\partial A}\frac\lambda2\mathrm{Tr}(A^\top GA)=\frac\lambda2(G+G^\top)A=\lambda GA$.
$$\boxed{\frac{\partial J}{\partial A}=\frac{H^\top D}{N}+\lambda GA,\qquad \frac{\partial J}{\partial b}=\frac{1}{N}\sum_i D_{i,:}}$$
The bias is not regularised (it only shifts all scores of one class; penalising it would bias the class priors).

**Mini-batch / stochastic version.** For a batch $B$: $g_A = H_B^\top D_B/|B| + \lambda GA$. Because the regulariser
does not depend on the data, $\mathbb E_B[g_A] = \partial J/\partial A$ (unbiased).

**Numerical gradient check.** `tests/test_gradients.py` compares the analytical gradients with central finite
differences at a random point whose margins are all at least $10^{-4}$ away from the kink - agreement to $10^{-6}$.

## 5. Efficient but exact SGD

Naive SGD needs $\lambda GA$ at every one of the $N$ updates per epoch ($O(M^2K)$ each). With $G = U\Lambda U^\top$
($U$ orthogonal) and $C = U^\top A$, $\tilde h_i = h_i U$:
$$A \leftarrow A - \eta(h_i^\top d_i + \lambda GA)\iff C \leftarrow (I-\eta\lambda\Lambda)C - \eta\,\tilde h_i^\top d_i ,$$
and the scores are $h_iA + b = \tilde h_i C + b$. Each update now costs $O(MK)$. Since the map $A\mapsto U^\top A$ is a
bijection that commutes with the update, the iterates are **identical**; the test
`test_sgd_one_sample_per_update_and_fast_path_equivalence` confirms agreement to $10^{-10}$. This trick is valid for
plain SGD/GD/MBGD (rotation invariant) but **not** for AdaGrad/Adam, whose per-coordinate scaling is not rotation
invariant - those are implemented directly in $A$-space.

## 6. Optimizer update rules (implemented manually)

| Optimizer | Batch | Update |
|---|---|---|
| GD | all $N$ | $\theta \leftarrow \theta - \eta g$ |
| SGD | 1 | $\theta \leftarrow \theta - \eta g_i$ |
| Mini-Batch GD | 64 | $\theta \leftarrow \theta - \eta g_B$ |
| AdaGrad | 64 | $s \leftarrow s + g_B^2$, $\theta \leftarrow \theta - \eta g_B/(\sqrt s + \epsilon)$ |
| Adam | 64 | $m \leftarrow \beta_1 m + (1-\beta_1)g$, $v \leftarrow \beta_2 v + (1-\beta_2)g^2$, $\theta \leftarrow \theta - \eta\,\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\epsilon}$ |

$\theta = (A, b)$ with separate optimizer states for $A$ and $b$; initialisation $A = 0$, $b = 0$ for every optimizer.

## 7. Conditioning - why plain GD is slow in coefficient space

For the hinge part, the curvature seen by GD in $A$-space is governed by $H^\top H/N$. RBF kernel columns are
positively correlated (all entries are in $(0,1]$), so $H^\top H/N$ has one very large eigenvalue (a "mean" direction)
and many tiny ones (measured: largest 41.9, median about 5e-5). Progress of plain GD along the many small-eigenvalue
directions is therefore extremely slow, and because the hinge loss is non-smooth, simply taking larger steps does not
help: the subgradient iterates start to oscillate (lr = 3 makes the objective blow up). Adaptive methods (AdaGrad, Adam) rescale each coordinate and are much less affected.
The measured eigenvalues are printed in notebook 01.

## 8. Relationship to other formulations

* **Binary SVM dual (libsvm / `SVC`)**: $\max_\alpha \sum\alpha_i - \frac12\sum\alpha_i\alpha_j y_iy_jk(x_i,x_j)$ subject to
  $0\le\alpha_i\le C$, $\sum\alpha_iy_i=0$, solved by SMO; multi-class via **One-vs-One** (15 binary problems for 6 classes).
* **This project**: unconstrained, primal, direct multi-class problem in RKHS coefficient space, optimised by
  first-order (sub)gradient methods. Same family of models (kernel expansion + hinge loss + RKHS norm) but a
  **different** optimisation problem - results are close but not identical.
* **Nystroem**: $\phi(x) = k(x,Z)U_r\Lambda_r^{-1/2}$. With the same basis, $A = U_r\Lambda_r^{-1/2}W$ gives
  $HA=\Phi W$ and $\mathrm{Tr}(A^\top GA)=\lVert W\rVert_F^2$: Mode C with Nystroem is a re-parametrisation of Mode B
  (verified in `tests/test_kernels.py`).
* **Random Fourier Features**: $\phi(x)=\sqrt{2/D}\cos(x\Omega+u)$, $\Omega_{\cdot j}\sim\mathcal N(0,2\gamma I)$,
  $u_j\sim U[0,2\pi]$ gives $\mathbb E[\phi(x)^\top\phi(z)] = e^{-\gamma\lVert x-z\rVert^2}$.

## 9. Complexity

| Quantity | Exact (M = N) | Reduced (M = 2,000) |
|---|---|---|
| Kernel matrix construction | $O(N^2 d)$ time, $O(N^2)$ memory | $O(NMd)$, $O(NM)$ |
| One GD step | $O(N^2K)$ | $O(NMK + M^2K)$ |
| One mini-batch step ($B$) | $O(BNK + N^2K)$ | $O(BMK + M^2K)$ |
| Prediction of one sample | $O(Nd + NK)$ | $O(Md + MK)$ |
