"""
Comprehensive evaluation metrics for medical image classification.

Includes:
  - Classification metrics: accuracy, F1, precision, recall.
  - Clinical metrics: sensitivity, specificity, PPV, NPV.
  - ROC-AUC (one-vs-rest, macro and per-class).
  - Average precision per class.
  - Statistical significance: paired t-test, Wilcoxon, Bonferroni.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)


def per_class_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: List[str] = None,
) -> pd.DataFrame:
    """Compute per-class sensitivity, specificity, PPV, NPV.

    Returns:
        DataFrame indexed by class name with columns
        [sensitivity, specificity, PPV, NPV, TP, FN, FP, TN].
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    rows = []
    for c in range(n_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sens = TP / (TP + FN) if (TP + FN) else 0.0
        spec = TN / (TN + FP) if (TN + FP) else 0.0
        ppv = TP / (TP + FP) if (TP + FP) else 0.0
        npv = TN / (TN + FN) if (TN + FN) else 0.0
        rows.append({
            "class": class_names[c],
            "sensitivity": sens,
            "specificity": spec,
            "PPV": ppv,
            "NPV": npv,
            "TP": int(TP), "FN": int(FN),
            "FP": int(FP), "TN": int(TN),
        })
    df = pd.DataFrame(rows).set_index("class")

    macro = df[["sensitivity", "specificity", "PPV", "NPV"]].mean()
    macro_row = pd.Series(
        {**macro.to_dict(), "TP": "-", "FN": "-", "FP": "-", "TN": "-"},
        name="macro_avg",
    )
    df = pd.concat([df, pd.DataFrame([macro_row])])
    return df


def compute_all_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
    class_names: List[str] = None,
) -> Dict:
    """Compute the full suite of metrics for a classification result."""
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # ROC-AUC (one-vs-rest, macro)
    try:
        y_one_hot = np.eye(n_classes)[y_true]
        auc_macro = roc_auc_score(y_one_hot, probs, average="macro", multi_class="ovr")
        auc_per_class = []
        for c in range(n_classes):
            yb = (y_true == c).astype(int)
            fpr, tpr, _ = roc_curve(yb, probs[:, c])
            auc_per_class.append(auc(fpr, tpr))
    except (ValueError, IndexError):
        auc_macro = float("nan")
        auc_per_class = [float("nan")] * n_classes

    # Average precision per class
    ap_per_class = []
    for c in range(n_classes):
        yb = (y_true == c).astype(int)
        ap_per_class.append(average_precision_score(yb, probs[:, c]))

    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    clinical = per_class_clinical_metrics(y_true, y_pred, n_classes, class_names)

    return {
        "accuracy": acc,
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "auc_macro_ovr": auc_macro,
        "auc_per_class": auc_per_class,
        "ap_per_class": ap_per_class,
        "confusion_matrix": cm.tolist(),
        "clinical_metrics": clinical.to_dict(),
    }


# ============================================================
# Statistical significance testing
# ============================================================
def paired_significance_tests(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
) -> Dict:
    """Paired t-test and Wilcoxon signed-rank test.

    Args:
        scores_a, scores_b: Per-fold scores (same length, paired by fold).

    Returns:
        Dict with t-statistic, t-p, Wilcoxon-statistic, W-p, mean diff,
        and 95% CI on the difference.
    """
    from scipy import stats

    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    diff = a - b
    n = len(diff)

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    se = stats.sem(diff)
    ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=se) if n > 1 else (np.nan, np.nan)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "n": n,
        "mean_diff": diff.mean(),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "t_stat": t_stat,
        "t_pvalue": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_p,
    }


def bonferroni_alpha(n_tests: int, alpha: float = 0.05) -> float:
    """Bonferroni-corrected significance threshold."""
    return alpha / max(n_tests, 1)


def pairwise_significance_matrix(
    runs: Dict[str, np.ndarray],
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Compute pairwise paired-t-test p-values for all model pairs.

    Args:
        runs: dict mapping model name -> array of per-fold scores.
        metric: name to use in output (just for column naming).

    Returns:
        Square DataFrame of p-values (NaN on diagonal).
    """
    names = list(runs.keys())
    n = len(names)
    pvals = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            res = paired_significance_tests(runs[a], runs[b], a, b)
            pvals.loc[a, b] = res["t_pvalue"]
    return pvals
