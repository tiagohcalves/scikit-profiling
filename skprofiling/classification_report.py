from typing import List, Optional
from itertools import cycle
import warnings

from IPython import display
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interp
from sklearn import metrics

from .utils import get_max_score, display_markdown

warnings.filterwarnings("ignore")

FIGSIZE_X = 10
FIGSIZE_Y = 8


def plot_pr_curve(y_true: List[float], y_pred: List[float]) -> None:
    pass


def plot_f1_th(y_true: List[float], y_pred: List[float]) -> None:
    pass


def plot_acc_th(y_true: List[float], y_pred: List[float]) -> None:
    pass


def plot_confusion_matrix(y_true: List[float], y_pred: List[float]) -> None:
    pass


def plot_calibration(y_true: List[float], y_pred: List[float]) -> None:
    pass


def plot_metrics_over_time(y_true: List[float], y_pred: List[float], date_index: List) -> None:
    pass


def classification_report(y_true: List[float], y_pred: List[float], date_index: Optional[list] = None) -> None:
    print_static_metrics(y_true, y_pred)

    display_markdown("# Plots")
    display_markdown("The plots shown here are meant to give insights about the distributions of the model")

    dist_score_plot(y_true, y_pred)
    plot_roc_curve(y_true, y_pred)
    plot_pr_curve(y_true, y_pred)
    plot_f1_th(y_true, y_pred)
    plot_acc_th(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_calibration(y_true, y_pred)

    if date_index is not None:
        plot_metrics_over_time(y_true, y_pred, date_index)


def print_static_metrics(y_true: List[float], y_pred: List[float]) -> None:
    display_markdown("# Statistical Metrics")
    display_markdown("These are statistical metrics that show some quality of the model.")
    display_markdown(
        "Since some of the metrics are threshold dependent, "
        "here is displayed the best value found by varying the threshold."
    )

    metrics_dict = {
        "Area Under ROC Curve": [metrics.roc_auc_score(y_true, y_pred), np.nan],
        "F1-Score": get_max_score(metrics.f1_score, y_true, y_pred),
        "Accuracy": get_max_score(metrics.accuracy_score, y_true, y_pred),
        "Average Precision-Recall": [metrics.average_precision_score(y_true, y_pred), np.nan],
        "Jaccard Score": get_max_score(metrics.jaccard_score, y_true, y_pred),
        "Humming Loss": get_max_score(metrics.accuracy_score, y_true, y_pred)
    }

    display_markdown("<br>")
    metrics_df = pd.DataFrame(metrics_dict, index=["Best Value", "Best Threshold"]).transpose()
    display.display(metrics_df)

    display_markdown("<br>")
    print(metrics.classification_report(y_true, np.round(y_pred)))
    display_markdown("---")


def dist_score_plot(y_true: List[float], y_pred: List[float]) -> None:
    display_markdown("## Score Distribution")
    display_markdown("This plot should give some clarity about how is the score distributed for each class.")
    display_markdown("Usually, the more separated are the scores, the best the model is.")
    display_markdown("Areas with high overlap represent non-confidant decisions.")

    # TODO: parametrize fig_size
    fig, ax = plt.subplots(figsize=(FIGSIZE_X, FIGSIZE_Y))

    for class_id in np.unique(y_true):
        ax = sns.distplot(y_pred[y_true == class_id], ax=ax, label=f"Class {class_id}")

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution Plot")
    ax.legend()

    plt.show()


def compute_roc(y_pred, y_true):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(np.unique(y_true)) - 1

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    return roc_auc, fpr, tpr


def plot_roc_curve(y_true, y_pred):
    display_markdown("## ROC Curve")

    if y_true.ndim == 1:
        roc_auc, fpr, tpr = compute_roc(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))
    else:
        roc_auc, fpr, tpr = compute_roc(y_pred, y_true)

    n_classes = len(np.unique(y_true)) - 1

    # Compute macro-average ROC curve and ROC area
    lw = 2

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
