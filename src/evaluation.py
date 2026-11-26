"""Evaluation metrics computed manually from the confusion matrix (verified against scikit-learn).

Important distinction:
  * the TRAINING OBJECTIVE is J = hinge + RKHS regularisation (what the optimizers minimise);
  * the EVALUATION METRICS (accuracy, precision, recall, F1, ROC-AUC) measure predictions.
Raw SVM scores are NOT probabilities, so no log-loss is reported. ROC-AUC is computed One-vs-Rest
directly from the decision scores, which is valid because AUC only depends on the ranking of scores.
"""
import numpy as np
import pandas as pd


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_prf(cm):
    """Precision, recall, F1 per class from the confusion matrix (rows = true, columns = predicted)."""
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, true_pos


def macro_f1(y_true, y_pred, n_classes):
    return float(per_class_prf(confusion_matrix(y_true, y_pred, n_classes))[2].mean())


def ovr_roc_auc(y_true, scores):
    """Macro One-vs-Rest ROC-AUC from raw decision scores (rank based, no probability assumption)."""
    from sklearn.metrics import roc_auc_score
    Y = np.eye(scores.shape[1])[np.asarray(y_true)]
    return float(roc_auc_score(Y, scores, average="macro"))


def evaluate(y_true, y_pred, n_classes, scores=None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    p, r, f, support = per_class_prf(cm)
    w = support / support.sum()
    out = {"accuracy": accuracy(y_true, y_pred),
           "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
           "weighted_precision": float(np.sum(w * p)), "weighted_recall": float(np.sum(w * r)),
           "weighted_f1": float(np.sum(w * f)),
           "per_class_precision": p, "per_class_recall": r, "per_class_f1": f, "support": support,
           "confusion_matrix": cm}
    if scores is not None:
        out["roc_auc_ovr_macro"] = ovr_roc_auc(y_true, scores)
    return out


def verify_with_sklearn(y_true, y_pred, metrics):
    """Max absolute difference between the manual metrics and scikit-learn's (should be ~1e-16)."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    diffs = [abs(metrics["accuracy"] - accuracy_score(y_true, y_pred))]
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        diffs += [abs(metrics[f"{avg}_precision"] - p), abs(metrics[f"{avg}_recall"] - r), abs(metrics[f"{avg}_f1"] - f)]
    return float(max(diffs))


def classification_report_df(metrics, class_names):
    df = pd.DataFrame({"precision": metrics["per_class_precision"], "recall": metrics["per_class_recall"],
                       "f1": metrics["per_class_f1"], "support": metrics["support"].astype(int)}, index=class_names)
    df.loc["macro avg"] = [metrics["macro_precision"], metrics["macro_recall"], metrics["macro_f1"], int(metrics["support"].sum())]
    df.loc["weighted avg"] = [metrics["weighted_precision"], metrics["weighted_recall"], metrics["weighted_f1"], int(metrics["support"].sum())]
    df["support"] = df["support"].astype(int)
    return df.round(4)
