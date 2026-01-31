from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr: float = 0.95) -> float:
    fpr, tpr_curve, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr_curve, tpr, side="left")
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, specificity: float = 0.95
) -> float:
    fpr, tpr_curve, _ = roc_curve(y_true, y_score)
    spec_curve = 1.0 - fpr
    mask = spec_curve >= specificity
    if np.any(mask):
        return float(tpr_curve[mask].max())
    return float(tpr_curve[np.argmax(spec_curve)])


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": compute_auroc(y_true, y_score),
        "auprc": compute_auprc(y_true, y_score),
        "fpr_at_95_tpr": fpr_at_tpr(y_true, y_score, tpr=0.95),
        "sens_at_95_spec": sensitivity_at_specificity(y_true, y_score, specificity=0.95),
    }


def evaluate_from_scores(score_csv: str, label_col: str = "label", score_col: str = "score") -> Dict[str, float]:
    df = pd.read_csv(score_csv)
    y_true = df[label_col].astype(int).to_numpy()
    y_score = df[score_col].astype(float).to_numpy()
    return evaluate_scores(y_true, y_score)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    metrics = {key: [] for key in ["auroc", "auprc", "fpr_at_95_tpr", "sens_at_95_spec"]}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        res = evaluate_scores(y_true[idx], y_score[idx])
        for key, val in res.items():
            metrics[key].append(val)
    return {
        key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for key, vals in metrics.items()
    }


# =============================================================================
# Calibration Metrics
# =============================================================================


def normalize_scores_to_probabilities(
    y_score: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize anomaly scores to [0, 1] range for calibration analysis.

    Args:
        y_score: Raw anomaly scores (higher = more anomalous)
        method: Normalization method ('minmax' or 'sigmoid')

    Returns:
        Normalized scores in [0, 1]
    """
    if method == "minmax":
        score_min, score_max = y_score.min(), y_score.max()
        if score_max > score_min:
            return (y_score - score_min) / (score_max - score_min)
        return np.zeros_like(y_score)
    elif method == "sigmoid":
        # Center around median and apply sigmoid
        centered = y_score - np.median(y_score)
        # Scale by IQR to get reasonable sigmoid range
        iqr = np.percentile(y_score, 75) - np.percentile(y_score, 25)
        if iqr > 0:
            centered = centered / iqr
        return 1.0 / (1.0 + np.exp(-centered))
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data for calibration visualization.

    Args:
        y_true: Binary labels (0 or 1)
        y_prob: Predicted probabilities in [0, 1]
        n_bins: Number of bins for the diagram

    Returns:
        Tuple of:
        - bin_edges: (n_bins + 1,) array of bin edges
        - bin_accuracies: (n_bins,) array of mean accuracy in each bin
        - bin_confidences: (n_bins,) array of mean confidence in each bin
        - bin_counts: (n_bins,) array of sample counts in each bin
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            # Include right edge for the last bin
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_accuracies[i] = y_true[mask].mean()
            bin_confidences[i] = y_prob[mask].mean()

    return bin_edges, bin_accuracies, bin_confidences, bin_counts


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy,
    weighted by the number of samples in each bin.

    Args:
        y_true: Binary labels (0 or 1)
        y_prob: Predicted probabilities in [0, 1]
        n_bins: Number of bins

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    _, bin_accuracies, bin_confidences, bin_counts = compute_reliability_diagram(
        y_true, y_prob, n_bins
    )

    total_samples = bin_counts.sum()
    if total_samples == 0:
        return 0.0

    ece = 0.0
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total_samples) * abs(
                bin_accuracies[i] - bin_confidences[i]
            )

    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum absolute difference between confidence and accuracy
    across all bins.

    Args:
        y_true: Binary labels (0 or 1)
        y_prob: Predicted probabilities in [0, 1]
        n_bins: Number of bins

    Returns:
        MCE value (lower is better)
    """
    _, bin_accuracies, bin_confidences, bin_counts = compute_reliability_diagram(
        y_true, y_prob, n_bins
    )

    mce = 0.0
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mce = max(mce, abs(bin_accuracies[i] - bin_confidences[i]))

    return float(mce)


def compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Compute Brier Score for probabilistic predictions.

    Brier Score = mean((y_prob - y_true)^2)
    Lower is better, 0 = perfect.

    Args:
        y_true: Binary labels (0 or 1)
        y_prob: Predicted probabilities in [0, 1]

    Returns:
        Brier score
    """
    return float(np.mean((y_prob - y_true) ** 2))


def evaluate_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    normalize_method: str = "minmax",
) -> Dict[str, float]:
    """
    Compute all calibration metrics.

    Args:
        y_true: Binary labels (0 or 1)
        y_score: Raw anomaly scores (will be normalized to [0, 1])
        n_bins: Number of bins for ECE/MCE
        normalize_method: Method to normalize scores ('minmax' or 'sigmoid')

    Returns:
        Dictionary with ECE, MCE, and Brier score
    """
    y_prob = normalize_scores_to_probabilities(y_score, method=normalize_method)

    return {
        "ece": compute_ece(y_true, y_prob, n_bins),
        "mce": compute_mce(y_true, y_prob, n_bins),
        "brier": compute_brier_score(y_true, y_prob),
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a reliability diagram for calibration visualization.

    Args:
        y_true: Binary labels (0 or 1)
        y_prob: Predicted probabilities in [0, 1]
        n_bins: Number of bins
        title: Plot title
        save_path: If provided, save the figure to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install via `pip install matplotlib`.")
        return

    bin_edges, bin_accuracies, bin_confidences, bin_counts = compute_reliability_diagram(
        y_true, y_prob, n_bins
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Main reliability diagram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)

    # Bar plot of accuracy per bin
    ax1.bar(
        bin_centers,
        bin_accuracies,
        width=bin_width * 0.8,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Model",
    )

    # Gap visualization (difference from perfect calibration)
    for i, (center, acc, conf, count) in enumerate(
        zip(bin_centers, bin_accuracies, bin_confidences, bin_counts)
    ):
        if count > 0:
            gap = abs(acc - conf)
            color = "red" if acc < conf else "green"
            ax1.plot([center, center], [conf, acc], color=color, linewidth=2, alpha=0.7)

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.bar(
        bin_centers,
        bin_counts,
        width=bin_width * 0.8,
        alpha=0.7,
        color="gray",
        edgecolor="black",
    )
    ax2.set_xlim([0, 1])
    ax2.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved reliability diagram to {save_path}")
    else:
        plt.show()

    plt.close()


def evaluate_scores_with_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    normalize_method: str = "minmax",
) -> Dict[str, float]:
    """
    Evaluate both discrimination and calibration metrics.

    Args:
        y_true: Binary labels (0 or 1)
        y_score: Raw anomaly scores
        n_bins: Number of bins for calibration metrics
        normalize_method: Method to normalize scores for calibration

    Returns:
        Dictionary with all metrics (AUROC, AUPRC, FPR@95%TPR, Sens@95%Spec, ECE, MCE, Brier)
    """
    # Discrimination metrics
    metrics = evaluate_scores(y_true, y_score)

    # Calibration metrics
    calibration = evaluate_calibration(y_true, y_score, n_bins, normalize_method)
    metrics.update(calibration)

    return metrics

