import warnings
from inspect import signature
from itertools import cycle
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from IPython import display
from matplotlib import pyplot as plt
from scipy import interp
from sklearn import metrics, calibration

from .utils import get_max_score, display_markdown

warnings.filterwarnings("ignore")

FIGSIZE_X = 10
FIGSIZE_Y = 8


def classification_report(y_true: List[float], y_pred: List[float], date_index: Optional[list] = None) -> None:
    print_static_metrics(y_true, y_pred)

    display_markdown("---")
    display_markdown("# Plots")
    display_markdown("The plots shown here are meant to give insights about the distributions of the model")

    dist_score_plot(y_true, y_pred)
    plot_roc_curve(y_true, y_pred)
    plot_pr_curve(y_true, y_pred)
    plot_th_impact(y_true, y_pred)
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
        "Area Under ROC Curve*": [metrics.roc_auc_score(y_true, y_pred), np.nan],
        "F1-Score": get_max_score(metrics.f1_score, y_true, y_pred),
        "Accuracy": get_max_score(metrics.accuracy_score, y_true, y_pred),
        "Average Precision-Recall*": [metrics.average_precision_score(y_true, y_pred), np.nan],
        "Jaccard Score": get_max_score(metrics.jaccard_score, y_true, y_pred),
        "Humming Loss": get_max_score(metrics.accuracy_score, y_true, y_pred)
    }

    display_markdown("<br>")
    metrics_df = pd.DataFrame(metrics_dict, index=["Best Value", "Best Threshold"]).transpose()
    display.display(metrics_df)

    display_markdown("<br>")
    print(metrics.classification_report(y_true, np.round(y_pred)))

    display_markdown("<br>")
    display_markdown("\\* These metrics do not depend on the threshold.")


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
    display_markdown("This curve indicates how many false positive one should expect before encountering a true "
                     "positive.")
    display_markdown("Usually, the bigger the area under the ROC curve, the better the model is predicting the target"
                     "class. ")
    display_markdown("A random model would have a straight line plot.")
    display_markdown("Also, the shape of the curve indicates if the model is making more mistakes on false positives"
                     "or false negatives.")

    if np.array(y_true).ndim == 1:
        roc_auc, fpr, tpr = compute_roc(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))
    else:
        roc_auc, fpr, tpr = compute_roc(y_pred, y_true)

    # From https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
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
    plt.figure(figsize=(FIGSIZE_X, FIGSIZE_Y))
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

    display_markdown("Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html")


def plot_pr_curve(y_true: List[float], y_pred: List[float]) -> None:
    display_markdown("## PR Curve")
    display_markdown("This plot shows the trade-off between precision and recall for different thresholds.")
    display_markdown("The bigger the area under the PR curve, the better the model is predicting the target"
                     "class. ")

    # From https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    average_precision = metrics.average_precision_score(y_true, y_pred)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.figure(figsize=(FIGSIZE_X, FIGSIZE_Y))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve. Average Precision = {0:0.2f}'.format(average_precision))
    plt.show()

    display_markdown("Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html")


def plot_th_impact(y_true: List[float], y_pred: List[float]) -> None:
    display_markdown("## Threshold Impact")
    display_markdown("This plot shows how the threshold impacts the model's performance")
    display_markdown("It indicates regions of threshold where the model is better separating the classes.")
    display_markdown("It should be useful when one is considering a sub-optimum threshold to meet some system"
                     "requirement.")

    f1_score, max_th, score_list_f1, thresholds = get_max_score(metrics.f1_score, y_true, y_pred, with_values=True)
    acc_score, max_th, score_list_acc, thresholds = get_max_score(metrics.accuracy_score, y_true, y_pred,
                                                                  with_values=True)

    plt.figure(figsize=(FIGSIZE_X, FIGSIZE_Y))
    plt.plot(thresholds, score_list_f1, label="F1-Score")
    plt.plot(thresholds, score_list_acc, label="Accuracy")

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Metrics by Threshold. Best F1-Score = {0:0.2f}'.format(f1_score))
    plt.legend()
    plt.show()

def plot_calibration(y_true: List[float], y_pred: List[float]) -> None:
    # TODO: Explain
    display_markdown("## Probability Calibration Curve")

    clf_score = metrics.brier_score_loss(y_true, y_pred, pos_label=max(y_true))
    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(y_true, y_pred, n_bins=10)

    fig, ax = plt.subplots(2, 1, figsize=(FIGSIZE_X, FIGSIZE_Y))

    ax[0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax[0].plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Brier Score: {clf_score:0.2f}")
    ax[1].hist(y_pred, range=(0, 1), bins=10, histtype="step", lw=2)

    ax[0].set_ylabel("Fraction of positives")
    ax[0].set_ylim([-0.05, 1.05])
    ax[0].legend(loc="lower right")
    ax[0].set_title('Calibration plots (reliability curve)')

    ax[1].set_xlabel("Mean predicted value")
    ax[1].set_ylabel("Count")
    # ax[1].legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

    display_markdown("Ref: https://scikit-learn.org/stable/auto_examples/calibration/"
                     "plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py")


def get_rolling_scores(metric_function, df, window_size):
    for i in range(window_size, df.shape[0]):
        yield metric_function(df.iloc[i-window_size:i]["true"], np.round(df.iloc[i-window_size:i]["pred"]))


def plot_metrics_over_time(y_true: List[float], y_pred: List[float], date_index: List) -> None:
    display_markdown("## Metrics Over Time")

    df = pd.DataFrame({"true": y_true, "pred": y_pred, "date": date_index})
    df.sort_values("date", inplace=True)

    window_size = min(df.shape[0], 10)
    rolling_acc = list(get_rolling_scores(metrics.accuracy_score, df, window_size))
    rolling_f1 = list(get_rolling_scores(metrics.f1_score, df, window_size))

    plt.figure(figsize=(FIGSIZE_X, FIGSIZE_Y))

    plt.plot(df["date"].iloc[window_size:], rolling_acc, label="Accuracy")
    plt.plot(df["date"].iloc[window_size:], rolling_f1, label="F1-Score")

    plt.xlabel("Date")
    plt.ylabel("Accuracy")
    plt.title("Metrics Over Time")
    plt.legend()

    plt.show()

