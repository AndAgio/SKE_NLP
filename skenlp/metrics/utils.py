import numpy as np


def format_predictions(y_true, y_predicted):
    if y_true.shape != y_predicted.shape:
        y_predicted = np.argmax(y_predicted, axis=1)

    return y_true, y_predicted
