from typing import List, Optional
from IPython import display

import numpy as np


def get_max_score(
        metric_function, y_true: List[float], y_pred: List[float],
        min_score: bool = False, with_values: bool = False
) -> (float, float, Optional[List[float]], Optional[List[float]]):
    thresholds = np.arange(0, 1, 0.01)
    score_list = []

    max_th = thresholds[0]
    max_score = np.inf if min_score else -np.inf

    for th in thresholds:
        try:
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
        except ValueError:
            pass

    if with_values:
        return max_score, max_th, score_list, thresholds

    return max_score, max_th


def display_markdown(text: str) -> None:
    display.display(display.Markdown(text))
