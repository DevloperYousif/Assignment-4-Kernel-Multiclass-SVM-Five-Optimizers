# Experiment Report

## 1. Setup
* Data: UCI HAR, official split. Training split 5,551 windows (16 subjects), validation 1,801 (5 subjects), test 2,947 (9 subjects).
* Preprocessing: standardisation fitted on the training split; subject ids never used as features.
* Kernel: RBF, gamma0 (median heuristic) = 1.0774e-03, gamma = 4 x gamma0 = 4.3096e-03; lambda = 0.0001.
* Representation: Mode B, M = 2,000 class-stratified basis samples (identical for all optimizers).
* Budget: 100 epochs; initial A = 0, b = 0; batch 64 for MBGD / AdaGrad / Adam, 1 for SGD, N for GD.
* Learning rate: 5 candidates per optimizer, each trained for the full budget; selection by final validation Macro F1.

## 2. Kernel configuration selection (validation, Adam lr 1e-3, 30 epochs)
Best: gamma = 4 x gamma0, lambda = 0.0001, validation Macro F1 0.9241.

| gamma / gamma0 | lambda=1e-05 | lambda=0.0001 | lambda=0.001 | lambda=0.01 |
|---|---|---|---|---|
| 0.5 | 0.7223 | 0.7885 | 0.7122 | 0.6034 |
| 1 | 0.8874 | 0.8911 | 0.8746 | 0.7016 |
| 2 | 0.9005 | 0.8888 | 0.8849 | 0.8232 |
| 4 | 0.9206 | 0.9241 | 0.9062 | 0.8544 |
| 8 | 0.9033 | 0.9047 | 0.8944 | 0.8531 |

## 3. Main result - five optimizers (test set)
| Optimizer | LR | Batch | Updates | Accuracy | Macro P | Macro R | Macro F1 | Val Macro F1 | Hinge | Reg | Total J | Train time (s) | Pred time (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gradient Descent | 1 | all | 100 | 86.46% | 0.8730 | 0.8589 | 0.8577 | 0.8561 | 0.4257 | 0.1513 | 0.5771 | 5.2 | 9.6 |
| SGD | 0.003 | 1 | 555,100 | 93.59% | 0.9343 | 0.9333 | 0.9333 | 0.9316 | 0.0993 | 0.0397 | 0.1391 | 11.3 | 9.3 |
| Mini-Batch GD | 0.1 | 64 | 8,700 | 92.37% | 0.9239 | 0.9206 | 0.9210 | 0.9296 | 0.1290 | 0.0384 | 0.1674 | 35.6 | 9.1 |
| AdaGrad | 0.1 | 64 | 8,700 | 93.82% | 0.9369 | 0.9366 | 0.9365 | 0.9405 | 0.1041 | 0.0402 | 0.1443 | 36.0 | 12.7 |
| Adam | 0.003 | 64 | 8,700 | 92.70% | 0.9300 | 0.9254 | 0.9252 | 0.9450 | 0.1135 | 0.0457 | 0.1592 | 37.9 | 14.6 |

Automatic verdict (notebook 05): {
 "best_by_validation_macro_f1": "Adam",
 "highest_test_accuracy": "AdaGrad",
 "lowest_training_objective": "SGD",
 "fastest_training": "GD",
 "convergence_target_J": 0.153,
 "epochs_to_target": {
  "GD": null,
  "SGD": 60,
  "MBGD": null,
  "AdaGrad": 66,
  "Adam": 49
 },
 "fastest_convergence_epochs": "Adam",
 "most_stable_epoch_level (lowest % epochs with J increase, then tail std)": "SGD",
 "best_efficiency (fastest within 1 Macro-F1 point of the best)": "SGD",
 "best_per_class_f1": {
  "WALKING": "Adam",
  "WALKING_UPSTAIRS": "Adam",
  "WALKING_DOWNSTAIRS": "Adam",
  "SITTING": "SGD",
  "STANDING": "SGD",
  "LAYING": "AdaGrad"
 }
}

### Per-class test F1
| Class | Gradient Descent | SGD | Mini-Batch GD | AdaGrad | Adam |
|---|---|---|---|---|---|
| WALKING | 0.9207 | 0.9474 | 0.9336 | 0.9521 | 0.9530 |
| WALKING_UPSTAIRS | 0.9125 | 0.9411 | 0.9386 | 0.9476 | 0.9554 |
| WALKING_DOWNSTAIRS | 0.7855 | 0.8750 | 0.8697 | 0.8956 | 0.9052 |
| SITTING | 0.7266 | 0.9140 | 0.8811 | 0.9068 | 0.8523 |
| STANDING | 0.8197 | 0.9291 | 0.9096 | 0.9225 | 0.8984 |
| LAYING | 0.9810 | 0.9935 | 0.9935 | 0.9944 | 0.9870 |

## 4. Basis size (Mode B vs exact Mode A)
| Basis M | Exact? | Val Macro F1 | Kernel time (s) | Train time, 30 epochs (s) | Kernel memory (MiB) |
|---|---|---|---|---|---|
| 250 | no | 0.8742 | 0.10 | 0.3 | 20 |
| 500 | no | 0.8877 | 0.17 | 1.2 | 41 |
| 1,000 | no | 0.9095 | 0.31 | 3.6 | 86 |
| 2,000 | no | 0.9241 | 0.64 | 12.2 | 188 |
| 4,000 | no | 0.9371 | 1.59 | 61.0 | 436 |
| 5,551 | yes (Mode A) | 0.9406 | 2.76 | 131.8 | 671 |

## 5. Kernel comparison
| Kernel | min eigenvalue of G | PSD | Val Macro F1 | Test accuracy | Test Macro F1 |
|---|---|---|---|---|---|
| linear | -1.84e-10 | yes | 0.8790 | 89.38% | 0.8891 |
| polynomial (d=2) | 0.00391 | yes | 0.9085 | 90.43% | 0.9001 |
| polynomial (d=3) | 0.0193 | yes | 0.9207 | 89.28% | 0.8889 |
| rbf | 0.0471 | yes | 0.9365 | 93.48% | 0.9330 |
| sigmoid [exploratory] | -49.2 | NO | 0.9444 | 92.98% | 0.9287 |

The linear kernel trained in coefficient space is poorly conditioned (objective 6.74 after 100 epochs); the same model in the
primal (controlled Linear SVM) is much better. The sigmoid kernel has a negative Gram eigenvalue and is reported only as exploratory.

## 6. Linear SVM vs Kernel SVM vs scikit-learn SVC
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

## 7. Exact-kernel verification and approximation quality
* A: exact (M = N): final J 0.1686, val Macro F1 0.8996
* B: restricted (M = 300): final J 0.3713, val Macro F1 0.8598
* C: Nystroem features (M = 300): final J 0.3724, val Macro F1 0.9085

| Dimension | Nystroem | Random Fourier Features |
|---|---|---|
| 250 | 0.1909 | 0.3481 |
| 500 | 0.1634 | 0.2480 |
| 1000 | 0.1414 | 0.1782 |
| 2000 | 0.1228 | 0.1265 |

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
