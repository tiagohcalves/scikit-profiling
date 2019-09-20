from typing import List, Optional
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interp
from sklearn import metrics


def get_max_score(
        metric_function, y_true: List[float], y_pred: List[float],
        min_score: bool = False, with_values: bool = False
) -> (float, float, Optional[List[float]], Optional[List[float]]):
    thresholds = np.arange(0, 1, 0.01)
    score_list = []

    max_th = thresholds[0]
    max_score = np.inf if min_score else -np.inf

    for th in thresholds:
        y_pred_label = np.where(y_pred < th, 0, 1)
        score_th = metric_function(y_true, y_pred_label)
        score_list.append(score_th)

        if (
                (min_score and (score_th < max_score))
                or
                (not min_score and score_th > max_score)
        ):
            max_score = score_th
            max_th = th

    if with_values:
        return max_score, max_th, score_list, thresholds

    return max_score, max_th


def classification_report(y_true: List[float], y_pred: List[float], date_index: list = None) -> None:
    print_static_metrics(y_pred, y_true)
    dist_score_plot(y_true, y_pred)


def print_static_metrics(y_true: List[float], y_pred: List[float]) -> None:
    metrics_dict = {
        "Area Under ROC Curve": [metrics.roc_auc_score(y_true, y_pred), np.nan],
        "F1-Score": get_max_score(metrics.f1_score, y_true, y_pred),
        "Accuracy": get_max_score(metrics.accuracy_score, y_true, y_pred),
        "Average Precision-Recall": [metrics.average_precision_score(y_true, y_pred), np.nan],
        "Jaccard Score": get_max_score(metrics.jaccard_similarity_score, y_true, y_pred),
        "Humming Loss": get_max_score(metrics.accuracy_score, y_true, y_pred)
    }

    metrics_df = pd.DataFrame(metrics_dict, index=["Best Value", "Best Threshold"]).transpose()
    print(metrics_df)
    print(metrics.classification_report(y_true, np.round(y_pred)))


def dist_score_plot(y_true: List[float], y_pred: List[float]) -> None:
    # TODO: allow multiclass
    fig, ax = plt.subplots()

    ax = sns.distplot(y_pred[y_true == 0], ax=ax)
    ax = sns.distplot(y_pred[y_true == 1], ax=ax)

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


def plot_roc_all(y_pred, y_true):
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
