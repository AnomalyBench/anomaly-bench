"""LIME explanation discovery module.

Note: this package was renamed `lime_method` to remove conflicts with the `lime` package.
"""
import os
import math
import time

import numpy as np
from lime import lime_tabular

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import forecasting_choices, reconstruction_choices
from explanation.explainers import Explainer, get_entropy
from explanation.lime_method.helpers import get_important_fts


class LIME(Explainer):
    """LIME explanation discovery method.
    """
    def __init__(self, args, output_path, explained_model, normal_model_samples):
        super().__init__(args, output_path, explained_model, normal_model_samples)
        # method-specific fixed parameters
        self.explainer = lime_tabular.RecurrentTabularExplainer(
            normal_model_samples, feature_names=[f'ft_{i}' for i in range(19)],
            mode='regression', discretize_continuous=True, discretizer='decile'
        )
        # method-specific hyperparameters
        self.model_w_size = args.window_size
        self.n_features = args.lime_n_features

    def fit_sample_parameters(self, sample, sample_labels):
        pass

    def fit_evaluate_sample(self, sample, sample_labels, test_prop=0.2, n_runs=5):
        """"""
        metrics = dict()
        # fit sample parts, feature clusters and false positive features
        self.fit_sample(sample, sample_labels)

        # get important features using the whole sample data (multiple windows for covering the whole anomaly)
        test_windows, test_start_ids = self.get_sample_windows(sample, extraction_purpose='coverage')
        test_important_fts = []
        for i in range(len(test_start_ids)):
            exp = self.explainer.explain_instance(
                test_windows[i:i+1], self.explained_model.score_windows, num_features=self.n_features
            )
            test_important_fts.extend(get_important_fts(exp))
        metrics['important_fts'] = list(set(test_important_fts))

        # get local stability using `n_runs` different test windows
        run_windows, run_start_ids = self.get_sample_windows(sample, extraction_purpose='stability', n_runs=n_runs)
        runs_important_fts = []
        start_time = time.time()
        for i in range(len(run_start_ids)):
            exp = self.explainer.explain_instance(
                run_windows[i:i+1], self.explained_model.score_windows, num_features=self.n_features
            )
            runs_important_fts.extend(get_important_fts(exp))
        metrics['fit_eval_time'] = time.time() - start_time
        metrics['local_stability'] = get_entropy(sample.shape[1], runs_important_fts)
        return metrics

    def get_sample_windows(self, sample, extraction_purpose, n_runs=None):
        a_t = 'windows must either be extracted for `coverage` or an experiment on `stability`'
        assert extraction_purpose in ['coverage', 'stability'], a_t
        anomaly_start_idx, anomaly_end_idx = len(self.normal_part), len(sample)
        if extraction_purpose == 'coverage':
            # number of test windows and their starting indices
            n_test_windows = math.floor((anomaly_end_idx - anomaly_start_idx) / self.model_w_size)
            window_start_ids = [
                anomaly_end_idx - self.model_w_size * (n_test_windows - w_i)
                for w_i in range(n_test_windows)
            ]
            if anomaly_end_idx >= self.model_w_size * (n_test_windows + 1):
                window_start_ids.insert(0, anomaly_end_idx - self.model_w_size * (n_test_windows + 1))
            else:
                window_start_ids.insert(0, 0)
            test_windows = []
            for start_idx in window_start_ids:
                test_windows.append(sample[start_idx:start_idx + self.model_w_size])
            return np.array(test_windows), window_start_ids
        else:
            assert n_runs is not None, '`n_runs` must be provided if extracting windows for stability experiment'
            start_idx = len(self.normal_part) - (self.model_w_size - 1)
            if start_idx < 0:
                start_idx = 0
            window_start_ids = []
            max_step_size = self.model_w_size // n_runs
            for i in range(n_runs):
                window_start_ids.append(
                    start_idx + i * (min(max_step_size, math.floor((anomaly_end_idx - anomaly_start_idx) / n_runs)))
                )
            test_windows = []
            for start_idx in window_start_ids:
                test_windows.append(sample[start_idx:start_idx+self.model_w_size])
            return np.array(test_windows), window_start_ids