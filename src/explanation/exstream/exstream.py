"""EXstream explanation discovery module.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from explanation.explainers import Explainer, get_list_entropy
from explanation.exstream.helpers import (
    classify_records, get_feature_clusters, get_fp_features, segmentation, segmentation_entropy_reward
)


class EXstream(Explainer):
    """EXstream explanation discovery method.

    See http://www.lix.polytechnique.fr/Labo/Yanlei.Diao/publications/xstream-edbt2017.pdf for details.
    """
    def __init__(self, args, output_path, explained_model=None, normal_model_samples=None):
        # this method never uses an explained model nor normal model samples
        super().__init__(args, output_path)
        # method-specific sample parameters
        self.feature_clusters = None
        self.fp_features = None
        # method-specific hyperparameters
        self.tolerance = args.exstream_tolerance
        self.correlation_threshold = args.exstream_correlation_threshold

    def fit_sample_parameters(self, sample, sample_labels):
        # get feature clusters from the whole sample
        self.feature_clusters = get_feature_clusters(pd.DataFrame(sample))
        # get false positive features from the normal part only
        self.fp_features = get_fp_features(self.normal_part)

    def fit_evaluate_sample(self, sample, sample_labels, test_prop=0.2, n_runs=5):
        metrics = dict()
        self.fit_sample(sample, sample_labels)

        # get important features using the whole sample data
        important_fts, _, _ = self.explain_records(self.normal_part, self.anomalous_part)
        metrics['important_fts'] = important_fts

        # get local stability and average classification performance using `n_runs` random splits
        start_time = time.time()
        runs_important_fts, classification_keys = [], ['accuracy', 'precision', 'recall', 'f1-score']
        for k in classification_keys:
            metrics[k] = []
        for _ in range(n_runs):
            # train/test split
            normal_train, normal_test, = train_test_split(self.normal_part, test_size=test_prop, random_state=None)
            anomalous_train, anomalous_test, = train_test_split(
                self.anomalous_part, test_size=test_prop, random_state=None
            )
            # important features, anomalous segments and importance scores
            important_fts, anomalous_ft_segments, importance_scores = \
                self.explain_records(normal_train, anomalous_train)
            runs_important_fts.append(important_fts)
            # classification performance using the explanations as rules
            y_true = np.concatenate([np.zeros(shape=len(normal_test)), np.ones(shape=len(anomalous_test))])
            y_pred = classify_records(
                np.concatenate([normal_test, anomalous_test]), important_fts, anomalous_ft_segments
            )
            metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            for k, v in zip(classification_keys[1:], [p, r, f]):
                metrics[k].append(v)
        # average time to fit and evaluate the explanation method
        metrics['fit_eval_time'] = (time.time() - start_time) / n_runs
        # average classification metrics across runs
        for k in classification_keys:
            metrics[k] = sum(metrics[k]) / len(metrics[k])
        # use important features across runs to compute the local stability
        metrics['local_stability'] = get_list_entropy(runs_important_fts, sample.shape[1])
        return metrics

    def explain_records(self, normal_records, anomalous_records):
        """Returns features that are important in telling apart normal from anomalous records.

        Also, returns the anomalous "segments" (intervals) and importance scores (rewards) for
        the corresponding features.


        Args:
            normal_records (ndarray): normal records of shape `(n_normal_records, n_features)`.
            anomalous_records (ndarray): anomalous records of shape `(n_anomalous_records, n_features)`.

        Returns:
            list, list, list: important features, anomalous segments and importance scores.
        """
        # rewards and anomalous value "segments" (i.e. intervals) for each feature
        feature_rewards = np.zeros(normal_records.shape[1])
        anomalous_segments = []
        for ft in range(normal_records.shape[1]):
            # get reference and anomalous univariate time series for the feature
            tsr, tsa = normal_records[:, ft], anomalous_records[:, ft]
            normal, anomalous, mixed, len_segment = segmentation(tsa, tsr)
            feature_rewards[ft] = segmentation_entropy_reward(tsa, tsr, normal, anomalous, mixed, len_segment)
            anomalous_segments.append([(x[0], x[1]) for x in anomalous] + [(x[0], x[1]) for x in mixed])

        # false positive filtering
        if len(self.fp_features) > 0:
            feature_rewards[self.fp_features] = 0

        # get sorted feature rewards
        sorted_ft_rewards = sorted(feature_rewards, reverse=True)
        # first order difference of sorted feature rewards
        diff_sorted_ft_rewards = [
            sorted_ft_rewards[i] - sorted_ft_rewards[i+1] for i in range(len(sorted_ft_rewards) - 1)
        ]
        # only keep the features before the sharp drop
        leap_features = list(np.argwhere(
            feature_rewards < sorted_ft_rewards[diff_sorted_ft_rewards.index(max(diff_sorted_ft_rewards))]
        )[:, 0])
        if len(leap_features) > 0:
            feature_rewards[leap_features] = 0

        # correlation clustering
        clusters_set = list(set(self.feature_clusters))
        clusters_representative = []
        for j in range(len(clusters_set)):
            cur_list = [i for i, e in enumerate(self.feature_clusters) if e == j + 1]
            clusters_representative.append(cur_list[feature_rewards[cur_list].argmax()])
        for i in range(len(feature_rewards)):
            if i not in clusters_representative:
                feature_rewards[i] = 0

        # get important features
        important_features = list(np.argwhere(feature_rewards != 0)[:, 0])
        important_fts = important_features.copy()

        return important_fts, \
            [anomalous_segments[i] for i in important_features], list(feature_rewards[important_features])
