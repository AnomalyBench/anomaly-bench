"""Supervised threshold selection module.

Gathers various supervised threshold selection methods.
"""
import os
import pickle
from abc import abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm

import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from detection.helpers import threshold_scores
from visualization.functions import plot_minimizer


def minimize_by_search(objective, bins, n_bin_trials):
    """Returns the 'best-effort' minimum and minimizer of the provided `objective` function.

    The returned minimum and minimizer correspond to the minimum objective found through trying
    `n_bin_trials` inputs linearly drawn between each bin interval.

    Args:
        objective (func): objective (scalar) function to minimize.
        bins (list): list of (start, end) intervals from which tried inputs are linearly drawn.
        n_bin_trials (int): number of inputs to try within each bin as a `linspace`.

    Returns:
        float, float: the best minimum and minimizer found at the end of the minimization process.
    """
    minimum, minimizer = float('inf'), None
    print('throughout search within equal-sized bins...')
    for bin_ in tqdm(bins):
        for t in np.linspace(*bin_, n_bin_trials):
            value = objective(t)
            if value < minimum:
                minimum = value
                minimizer = t
    print('done.')
    return minimum, minimizer


class SupervisedSelector:
    """Base SupervisedSelector class.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the computed threshold and information to.
    """
    def __init__(self, args, output_path):
        self.output_path = output_path
        self.threshold = None

    def fit(self, periods_scores, periods_labels, evaluator):
        """Selects and saves the threshold maximizing the anomaly detection performance.

        Args:
            periods_scores (ndarray): record-wise outlier scores of the periods.
                Shape `(n_periods, period_length)`, where `period_length` depends on the period.
            periods_labels (ndarray): corresponding record-wise binary labels of the same shape.
            evaluator (metrics.Evaluator): object defining the binary metrics to optimize.
        """
        self.select_threshold(periods_scores, periods_labels, evaluator)
        # save the computed threshold to the output path
        print(f'saving selected threshold under {self.output_path}...', end=' ', flush=True)
        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.output_path, 'threshold.pkl'), 'wb') as pickle_file:
            pickle.dump(self.threshold, pickle_file)
        print('done.')

    def select_threshold(self, periods_scores, periods_labels, evaluator):
        """Performs the actual outlier score threshold selection.
        """
        # find the threshold maximizing the F-score
        flattened_scores = np.concatenate(periods_scores, axis=0)
        min_score, max_score = np.min(flattened_scores), np.max(flattened_scores)

        # the objective function to minimize is the negative average F-score
        def objective(threshold):
            periods_preds = threshold_scores(periods_scores, threshold)
            f_scores_dict, _, _ = evaluator.compute_metrics(periods_labels, periods_preds)
            return -f_scores_dict['avg']
        self.threshold = self.get_minimizer(flattened_scores, objective, min_score, max_score)
        plot_minimizer(objective, [min_score, max_score], self.threshold, output_path=self.output_path)

    @abstractmethod
    def get_minimizer(self, flattened_scores, objective, min_score, max_score):
        """Returns the score in `flattened_scores` minimizing `objective`.

        Args:
            flattened_scores (ndarray): 1d array of flattened record-wise outlier scores.
            objective (func): scalar function to minimize, taking as input a threshold
                value returning the corresponding performance of interest (e.g. F1-score).
            min_score (float): smallest score in `flattened_scores`.
            max_score (float): largest score in `flattened_scores`.

        Returns:
            float: the threshold value found as minimizing `objective`.
        """


class BrentSelector(SupervisedSelector):
    def __init__(self, args, output_path):
        super().__init__(args, output_path)

    def get_minimizer(self, flattened_scores, objective, min_score, max_score):
        """Returns the minimizer using Brent's algorithm."""
        return minimize_scalar(objective, bounds=[min_score, max_score], method='bounded').x


class SearchDetector(SupervisedSelector):
    def __init__(self, args, output_path):
        super().__init__(args, output_path)
        # number of bins and number of trials per bin
        self.n_bins = args.n_bins
        self.n_bin_trials = args.n_bin_trials

    def get_minimizer(self, flattened_scores, objective, min_score, max_score):
        """Returns the minimizer using linear searches inside equi-depth bins."""
        quantiles = np.quantile(flattened_scores, q=np.linspace(0, 1, self.n_bins + 1))
        bins = [(quantiles[i], quantiles[i+1]) for i in range(len(quantiles) - 1)]
        return minimize_by_search(objective, bins, self.n_bin_trials)[1]


class SmallestSelector(SupervisedSelector):
    def __init__(self, args, output_path):
        super().__init__(args, output_path)

    def get_minimizer(self, flattened_scores, objective, min_score, max_score):
        """Returns the smallest score in `flattened_scores`."""
        return min_score


class LargestSelector(SupervisedSelector):
    def __init__(self, args, output_path):
        super().__init__(args, output_path)

    def get_minimizer(self, flattened_scores, objective, min_score, max_score):
        """Returns the largest score in `flattened_scores`."""
        return max_score


# dictionary gathering references to the defined supervised threshold selection methods
selector_classes = {
    'smallest': SmallestSelector,
    'largest': LargestSelector,
    'brent': BrentSelector,
    'search': SearchDetector,
}
