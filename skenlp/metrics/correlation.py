import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(a: np.array, b: np.array) -> float:
    return dot(a, b)/(norm(a)*norm(b))


def pearson_corr(a: np.array, b: np.array) -> float:
    return np.corrcoef(a, b)[0, 1]


def dist(a: np.array, b: np.array) -> float:
    return np.linalg.norm(a-b)
