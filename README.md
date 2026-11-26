# Assignment 4 - Kernel-Based Multi-Class SVM with Five Optimizers

**Human Activity Recognition Using Kernel-Based Multi-Class Support Vector Machines and Five Optimization Algorithms**

A genuine kernelized, direct multi-class SVM (Crammer-Singer hinge loss + RKHS regularisation, RBF kernel) implemented
from scratch in NumPy and trained with five manually implemented optimizers - **Gradient Descent, SGD, Mini-Batch GD,
AdaGrad and Adam** - on the UCI HAR smartphone dataset (6 activities). One notebook per optimizer.

## Kaggle notebooks (executed on Kaggle)
1. [01 - Gradient Descent](https://www.kaggle.com/code/yousifotoom12/assignment-4-01-gd-kernel-svm)
2. [02 - SGD](https://www.kaggle.com/code/yousifotoom12/assignment-4-02-sgd-kernel-svm)
3. [03 - Mini-Batch GD](https://www.kaggle.com/code/yousifotoom12/assignment-4-03-minibatch-kernel-svm)
4. [04 - AdaGrad](https://www.kaggle.com/code/yousifotoom12/assignment-4-04-adagrad-kernel-svm)
5. [05 - Adam](https://www.kaggle.com/code/yousifotoom12/assignment-4-05-adam-kernel-svm)

## Results (independent test set: 2,947 windows from 9 unseen subjects)

| Optimizer | LR | Batch | Updates | Accuracy | Macro P | Macro R | Macro F1 | Val Macro F1 | Hinge | Reg | Total J | Train time (s) | Pred time (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gradient Descent | 1 | all | 100 | 86.46% | 0.8730 | 0.8589 | 0.8577 | 0.8561 | 0.4257 | 0.1513 | 0.5771 | 5.2 | 9.6 |
| SGD | 0.003 | 1 | 555,100 | 93.59% | 0.9343 | 0.9333 | 0.9333 | 0.9316 | 0.0993 | 0.0397 | 0.1391 | 11.3 | 9.3 |
| Mini-Batch GD | 0.1 | 64 | 8,700 | 92.37% | 0.9239 | 0.9206 | 0.9210 | 0.9296 | 0.1290 | 0.0384 | 0.1674 | 35.6 | 9.1 |
| AdaGrad | 0.1 | 64 | 8,700 | 93.82% | 0.9369 | 0.9366 | 0.9365 | 0.9405 | 0.1041 | 0.0402 | 0.1443 | 36.0 | 12.7 |
| Adam | 0.003 | 64 | 8,700 | 92.70% | 0.9300 | 0.9254 | 0.9252 | 0.9450 | 0.1135 | 0.0457 | 0.1592 | 37.9 | 14.6 |

* **Best optimizer by validation Macro F1 (selection rule):** Adam (0.9450)
* **Highest test accuracy:** AdaGrad (93.82%) · **highest test Macro F1:** AdaGrad (0.9365)
* **Lowest training objective:** SGD · **fastest training:** Gradient Descent (only 100 updates)
* **Linear SVM vs Kernel SVM:** Linear 94.91% / Macro F1 0.9486 vs RBF kernel (Kernel SVM (RBF, Adam)) 92.70% / 0.9252;
  scikit-learn `SVC` reference 94.60% / 0.9441. On this dataset the kernel is **not** better than the linear model.

Training times are from one run on a 2-vCPU machine and vary between runs; all metrics are deterministic.

## Dataset
UCI *Human Activity Recognition Using Smartphones* (Anguita et al., ESANN 2013) - 10,299 windows, 561 features,
30 volunteers, official subject-disjoint split 7,352 / 2,947. Download and placement: [`data/README.md`](data/README.md).
Kaggle mirror: `uciml/human-activity-recognition-with-smartphones` (the loader reads both formats).

## Method in one paragraph
Standardised features (fitted on training data) are compared with a class-stratified **kernel basis of M = 2,000 training
samples** through the RBF kernel $k(x,z)=\exp(-\gamma\lVert x-z\rVert^2)$, $\gamma = 4\times$ median heuristic
$= 4.310e-03$. Scores are $S = HA + b$ and the objective is
$J = \frac\lambda2\mathrm{Tr}(A^\top GA) + \frac1N\sum_i\max(0, 1+\max_{j\ne y_i}S_{ij}-S_{iy_i})$ with $\lambda = 10^{-4}$.
Subgradients $\partial J/\partial A = H^\top D/N + \lambda GA$, $\partial J/\partial b = \sum_i D_i/N$. Every optimizer starts from
$A = 0, b = 0$ and runs 100 epochs; its learning rate is chosen from 5 values by validation Macro F1. Validation subjects are
separated from training subjects with `GroupShuffleSplit` (seed 42). Details: [`documentation/Mathematical_Derivation.md`](documentation/Mathematical_Derivation.md).

**Representation mode:** Mode B (restricted basis, M = 2,000) for all optimizers - chosen from the measured resource report:
the exact basis (M = N = 5,551) fits in memory but makes each adaptive update 7.7x more expensive. The basis-size study
measures the cost of the restriction (val Macro F1 0.9241 at M = 2,000 vs 0.9406 exact).
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
├── models/                         kernel_svm_{gd,sgd,minibatch,adagrad,adam}.npz (A, b, basis, gamma, scaler, labels)
├── presentation/                   Assignment_4_Kernel_SVM.pptx (30 slides, Arabic speaker notes)
├── documentation/                  Academic_Defense_Guide.md · Mathematical_Derivation.md · Kernel_Theory.md · Experiment_Report.md
├── tests/                          test_kernels · test_kernel_svm · test_gradients · test_optimizers (pytest)
└── tools/                          notebook / presentation / documentation builders
```

## Installation and execution
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python data/rebuild_dataset.py     # rebuild the dataset from the split archive in data/
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
| Kernel | min eigenvalue of G | PSD | Val Macro F1 | Test accuracy | Test Macro F1 |
|---|---|---|---|---|---|
| linear | -1.84e-10 | yes | 0.8790 | 89.38% | 0.8891 |
| polynomial (d=2) | 0.00391 | yes | 0.9085 | 90.43% | 0.9001 |
| polynomial (d=3) | 0.0193 | yes | 0.9207 | 89.28% | 0.8889 |
| rbf | 0.0471 | yes | 0.9365 | 93.48% | 0.9330 |
| sigmoid [exploratory] | -49.2 | NO | 0.9444 | 92.98% | 0.9287 |

## Linear SVM vs Kernel SVM
| Metric | Linear SVM | Kernel SVM (RBF, Adam) | sklearn SVC (RBF, OvO, reference) |
|---|---|---|---|
| Accuracy | 94.91% | 92.70% | 94.60% |
| Macro Precision | 0.9500 | 0.9300 | 0.9446 |
| Macro Recall | 0.9479 | 0.9254 | 0.9440 |
| Macro F1 | 0.9486 | 0.9252 | 0.9441 |
| Training Time (s) | 0.918 | 41.034 | 2.192 |
| Prediction Time (s) | 0.003 | 0.211 | 2.512 |
| Model Complexity | 3,372 weights | 12,006 coefficients + 2,000 x 561 basis | 2,445 support vectors |
| Decision Boundary | Linear | Non-linear | Non-linear |
| Model Memory | 0.026 MiB | 8.65 MiB | - |

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
