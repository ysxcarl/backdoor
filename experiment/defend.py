
import numpy as np
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
import scipy.stats


# Simple element-wise median
def median(g_deltas):

    return np.median(g_deltas, axis=0)

# Beta is the proportion to trim from the top and bottom.
def trimmed_mean(g_deltas, beta):

    return scipy.stats.trim_mean(g_deltas, beta, axis=0)


# Returns the index of the row that should be used in Krum
def krum(deltas, n_client, clip):

    # assume deltas is an array of size group * d
    n = len(deltas)
    scores = get_krum_scores(deltas, n_client - clip)
    good_idx = np.argpartition(scores, n_client - clip)[:(n_client - clip)]

    return np.mean(deltas[good_idx], axis=0)


def get_krum_scores(X, groupsize):

    krum_scores = np.zeros(len(X))

    # Calculate distances
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(
        X**2, axis=1)[None] - 2 * np.dot(X, X.T)

    for i in range(len(X)):
        krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

    return krum_scores

# Simple element-wise mean
def average(g_deltas):
    return np.mean(g_deltas, axis=0)
