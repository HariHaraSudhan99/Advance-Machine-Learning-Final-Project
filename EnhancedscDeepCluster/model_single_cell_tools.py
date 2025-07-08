import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import linear_sum_assignment

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in ind) * 1.0 / y_pred.size

def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=0.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6, 3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    """
    Select highly variable genes based on dropout rate and expression.
    """
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.full_like(zeroRate, np.nan)
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (1 - zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.full_like(zeroRate, np.nan)
        detected = zeroRate < 1
        mask = data[:, detected] > threshold
        logs = np.full_like(data[:, detected], np.nan)
        logs[mask] = np.log2(data[:, detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up, low = 10, 0
        for _ in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = zeroRate > np.exp(-decay * (meanExpr - xoffset)) + yoffset
            selected = selected & nonan
            count = np.sum(selected)
            if count == n:
                break
            elif count < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        if verbose > 0:
            print(f'Chosen offset: {xoffset:.2f}')
    else:
        nonan = ~np.isnan(zeroRate)
        selected = zeroRate > np.exp(-decay * (meanExpr - xoffset)) + yoffset
        selected = selected & nonan

    if plot:
        if figsize:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        xmax = np.ceil(np.nanmax(meanExpr))
        plt.xlim([np.log2(threshold) if threshold > 0 else 0, xmax])

        x = np.arange(plt.xlim()[0], plt.xlim()[1] + 0.1, 0.1)
        y = np.exp(-decay * (x - xoffset)) + yoffset

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.vstack([np.column_stack([x, y]), [xmax, 1]])
        plt.gca().add_patch(plt.Polygon(xy, color=sns.color_palette()[1], alpha=0.4))

        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        plt.xlabel('Mean log2 nonzero expression')
        plt.ylabel('Frequency of zero expression' if threshold == 0 else 'Frequency of near-zero expression')

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0)] * len(markers)
            for idx, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[idx]
                plt.text(meanExpr[i] + dx + 0.1, zeroRate[i] + dy, g, color='k', fontsize=labelsize)

        plt.tight_layout()

    return selected