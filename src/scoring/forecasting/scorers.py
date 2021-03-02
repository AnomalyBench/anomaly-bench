import os
import pickle
from abc import abstractmethod

import numpy as np
from scipy.stats import multivariate_normal

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.forecasting.helpers import get_period_sequence_target_pairs


class ForecastingScorer:
    """Forecasting-based score assignment base class.

    Derives record-wise outlier scores from the predictions of a trained Forecaster object.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        forecaster (Forecaster): trained Forecaster object whose predictions are used to derive outlier scores.
        output_path (str): path to save the scoring model and information to.
        model (misc|None): if not None, the scorer will be initialized using the provided model.
    """
    def __init__(self, args, forecaster, output_path, model=None):
        self.forecaster = forecaster
        self.output_path = output_path
        self.model = model

    @classmethod
    def from_file(cls, args, forecaster, full_model_path):
        """Returns a ForecastingScorer object with its parameters initialized from a pickled model.

        Args:
            args (argparse.Namespace): parsed command-line arguments.
            forecaster (Forecaster): trained Forecaster object whose predictions are used to derive outlier scores.
            full_model_path (str): full path to the pickle model file.

        Returns:
            ForecastingScorer: pre-initialized ForecastingScorer object.
        """
        print(f'loading scoring model file {full_model_path}...', end=' ', flush=True)
        model = pickle.load(open(full_model_path, 'rb'))
        print('done.')
        return cls(args, forecaster, '', model)

    @abstractmethod
    def score_pairs(self, X, y):
        """Returns the outlier scores for the provided `(X, y)` (sequence, target) pairs.

        Args:
            X (ndarray): shape `(n_samples, n_back, n_features)`.
            y (ndarray): shape `(n_samples, n_features)` or `(n_samples, n_forward, n_features)`.

        Returns:
            ndarray: outlier scores for each pair, of shape `(n_pairs,)`.
        """

    def score_period(self, period):
        """Returns the record-wise outlier scores for the provided contiguous period.

        Args:
            period (ndarray): shape `(period_length, n_features)`.

        Returns:
            ndarray: shape `(period_length - n_back,)`. Where `n_back` is the number of records used to forecast.
        """
        return self.score_pairs(
            *get_period_sequence_target_pairs(period, self.forecaster.n_back, self.forecaster.n_forward)
        )

    def score(self, periods):
        """Returns the record-wise outlier scores for the provided periods.

        For more efficiency, we first concatenate the periods' (sequence, target) pairs, score them,
        and then separate back the periods scores.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`; `period_length` depends on the period.

        Returns:
            ndarray: shape `(n_periods, period_length - n_back,)`.
        """
        sequences, targets = [], []
        n_pairs = []
        print('creating and concatenating (sequence, target) pairs of the periods...', end=' ', flush=True)
        for period in periods:
            p_s, p_t = get_period_sequence_target_pairs(period, self.forecaster.n_back, self.forecaster.n_forward)
            sequences.append(p_s)
            targets.append(p_t)
            n_pairs.append(p_s.shape[0])
        sequences = np.concatenate(sequences, axis=0).astype(np.float64)
        targets = np.concatenate(targets, axis=0).astype(np.float64)
        print('done.')
        print('computing outlier scores for the pairs...', end=' ', flush=True)
        scores = self.score_pairs(sequences, targets)
        print('done.')
        periods_scores, cursor = [], 0
        print('grouping back scores by period...', end=' ', flush=True)
        for np_ in n_pairs:
            periods_scores.append(scores[cursor:cursor+np_])
            cursor += np_
        print('done.')
        return np.array(periods_scores, dtype=object)


class RelativeErrorScorer(ForecastingScorer):
    """Relative error-based method (non-parametric).

    The outlier score of a record is set to its relative error by the model.
    """
    def __init__(self, args, forecaster, output_path, model=None):
        super().__init__(args, forecaster, output_path, model)

    def score_pairs(self, X, y):
        """The scores are the relative forecasting errors.
        """
        y_pred = self.forecaster.predict(X)
        return np.linalg.norm(y - y_pred, axis=1) / np.linalg.norm(y)


class GaussianNLLScorer(ForecastingScorer):
    """Gaussian negative log-likelihood method.

    The outlier score of a record is set to the negative log-likelihood of its error vector with respect
    to the distribution of error vectors committed on a validation set. The validation error vectors are assumed
    to a follow a multivariate Gaussian distribution.
    """
    def __init__(self, args, forecaster, output_path, model=None):
        super().__init__(args, forecaster, output_path, model)

    def fit(self, X, y):
        """Fits the multivariate Gaussian to the forecaster's error vectors on the provided (sequence, target) pairs.

        Args:
            X (ndarray): sequences of shape `(n_pairs, n_back, n_features)`.
            y (ndarray): targets of shape `(n_pairs, n_features)` or `(n_pairs, n_forward, n_features)`.
        """
        print('fitting multivariate Gaussian to error vectors of the validation set...', end=' ', flush=True)
        errors = y - self.forecaster.predict(X)
        self.model = multivariate_normal(np.mean(errors, axis=0), np.cov(errors, rowvar=False))
        print('done.')
        print(f'saving fit distribution under {self.output_path}...', end=' ', flush=True)
        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.output_path, 'model.pkl'), 'wb') as pickle_file:
            pickle.dump(self.model, pickle_file)
        print('done.')

    def score_pairs(self, X, y):
        """The scores are the negative log-likelihood of error vectors w.r.t to the fit distribution.
        """
        y_pred = self.forecaster.predict(X)
        return -np.apply_along_axis(self.model.logpdf, arr=(y - y_pred), axis=1)


# dictionary gathering references to the defined scoring methods
scoring_classes = {
    're': RelativeErrorScorer,
    'nll': GaussianNLLScorer
}
