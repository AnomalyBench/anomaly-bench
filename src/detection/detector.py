import os
import pickle
import importlib

import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from scoring.forecasting.scorers import ForecastingScorer
from scoring.reconstruction.scorers import ReconstructionScorer
from detection.helpers import threshold_scores


class Detector:
    """Final anomaly detection class.

    Assigns binary anomaly predictions to the records of its contiguous periods based on
    its scorer and threshold.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        scorer (Scorer): object assigning record-wise outlier scores to contiguous periods.
        output_path (str): path to save the threshold and information to.
        threshold (float): if not None, the detector will be initialized using the provided
            threshold value.
    """
    def __init__(self, args, scorer, output_path, threshold=None):
        # object performing the outlier score threshold selection
        self.threshold_selector = importlib.import_module(
            f'detection.thresholding.{args.threshold_supervision}_selectors'
        ).selector_classes[args.threshold_selection](args, output_path)
        if threshold:
            self.threshold_selector.threshold = threshold
        # object performing the outlier score assignments
        self.scorer = scorer

    @classmethod
    def from_file(cls, args, scorer, full_threshold_path):
        """Returns a Detector object with its threshold initialized from an existing file.

        Args:
            args (argparse.Namespace): parsed command-line arguments.
            scorer (Scorer): object assigning record-wise outlier scores to contiguous periods.
            full_threshold_path (str): full path to the threshold pickle file.

        Returns:
            Detector: pre-initialized Detector object.
        """
        print(f'loading threshold file {full_threshold_path}...', end=' ', flush=True)
        threshold = pickle.load(open(full_threshold_path, 'rb'))
        print('done.')
        return cls(args, scorer, '', threshold)

    def fit(self, *, periods=None, periods_labels=None, evaluator=None, X=None, y=None):
        """Fits the detector's outlier score threshold.

        The threshold selection can either be `supervised` or `unsupervised`.
        - Supervised: the threshold is selected as maximizing the detection performance
            on `periods`, based on `periods_labels` and `evaluator`.
        - Unsupervised: the threshold is selected using the `(X[, y])` samples
            distribution.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                Where `period_length` depends on the period.
            periods_labels (ndarray): binary record-wise labels for the periods
                (already trimmed if needed).
            evaluator (metrics.Evaluator): object defining the binary metrics to optimize.
            X (ndarray): sequences of shape `(n_pairs, n_back, n_features)` or windows of
                shape `(n_windows, window_size, n_features)`.
            y (ndarray): targets of shape `(n_pairs, n_features)` or
                `(n_pairs, n_forward, n_features)`. Only relevant to forecasting models.
        """
        # supervised threshold selection
        if not (periods is None or periods_labels is None or evaluator is None):
            self.threshold_selector.fit(self.scorer.score(periods), periods_labels, evaluator)
        # unsupervised threshold selection
        else:
            scores = None
            if isinstance(self.scorer, ForecastingScorer):
                a_t = '(sequence, target) pairs have to be provided for unsupervised ts selection'
                assert not (X is None or y is None), a_t
                scores = self.scorer.score_pairs(X, y)
            if isinstance(self.scorer, ReconstructionScorer):
                a_t = 'window samples have to be provided for unsupervised ts selection'
                assert X is not None and y is None, a_t
                scores = self.scorer.score_windows(X)
            assert scores is not None, 'scorer must be either `ForecastingScorer` or `ReconstructionScorer`'
            self.threshold_selector.fit(scores)

    def predict_period(self, period):
        """Returns record-wise binary predictions for the provided period.

        TODO - There might be something to fix here.

        Args:
            period (ndarray): shape `(period_length, n_features)`.

        Returns:
            ndarray: predictions for the period, whose shape depends on the scorer type.
                - Forecasting-based scorers: shape `(period_length - n_back,)`.
                    Where `n_back` is the number of records used to forecast.
                - Reconstruction-based scorers: shape `(period_length,)`.
        """
        return threshold_scores(self.scorer.score_period(period), self.threshold_selector.threshold)

    def predict(self, periods):
        """Returns record-wise binary predictions for the provided periods.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                Where `period_length` depends on the period.

        Returns:
            ndarray: predictions for the period, whose shape depends on the scorer type.
                - Forecasting-based scorers: shape `(n_periods, period_length - n_back)`.
                    Where `n_back` is the number of records used to forecast.
                - Reconstruction-based scorers: shape `(n_periods, period_length)`.
        """
        return threshold_scores(self.scorer.score(periods), self.threshold_selector.threshold)
