"""Module gathering various explanation discovery methods to compare.
"""
import os
from abc import abstractmethod

import numpy as np
from scipy.stats import entropy


def get_entropy(count, value_list):
    freq = []
    for i in range(count):
        if value_list.count(i) != 0:
            freq.append(value_list.count(i)/len(value_list))
    return entropy(freq, base=2)


class Explainer:
    def __init__(self, args, output_path, explained_model=None, normal_model_samples=None):
        self.output_path = output_path
        self.explained_model = explained_model
        self.normal_model_samples = normal_model_samples
        # parameters fit to each sample
        self.normal_part, self.anomalous_part = None, None

    def fit_sample(self, sample, sample_labels):
        anomaly_start, anomaly_end = np.where(sample_labels > 0)[0][0], len(sample_labels)
        self.normal_part, self.anomalous_part = sample[:anomaly_start], sample[anomaly_start:]
        self.fit_sample_parameters(sample, sample_labels)

    def fit_evaluate_samples(self, samples, samples_labels, test_prop=0.2, n_runs=5):
        metrics_dict = dict()
        n_features = samples[0].shape[1]
        for sample, sample_labels in zip(samples, samples_labels):
            m_dict = self.fit_evaluate_sample(sample, sample_labels, test_prop, n_runs)
            for k in m_dict:
                if k in metrics_dict:
                    metrics_dict[k].append(m_dict[k])
                else:
                    metrics_dict[k] = [m_dict[k]]
        # compute samples consistency and entropy based on important features
        important_fts = metrics_dict.pop('important_fts')
        n_important_fts = [len(i_fts) for i_fts in important_fts]
        metrics_dict['consistency'] = sum(n_important_fts) / len(n_important_fts)
        metrics_dict['entropy'] = get_entropy(n_features, important_fts)
        # average the rest of the metrics across samples
        for m_name in [k for k in metrics_dict if k not in ['consistency', 'entropy']]:
            metrics_dict[m_name] = sum(metrics_dict[m_name]) / len(metrics_dict[m_name])
        # if classification metrics are not returned, fill them with NaNs
        for k in ['accuracy', 'precision', 'recall', 'f1-score']:
            if k not in metrics_dict:
                metrics_dict[k] = np.nan
        return metrics_dict

    @abstractmethod
    def fit_sample_parameters(self, sample, sample_labels):
        """"""

    @abstractmethod
    def fit_evaluate_sample(self, sample, sample_labels, test_prop=0.2, n_runs=5):
        """"""


# use a getter function to access references to explanation discovery classes to solve cross-import issues
def get_explanation_classes():
    """Returns a dictionary gathering references to the defined explanation discovery classes.
    """
    # add absolute src directory to python path to import other project modules
    import sys
    src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    sys.path.append(src_path)
    from explanation.exstream.exstream import EXstream
    from explanation.macrobase.macrobase import MacroBase
    from explanation.lime_method.lime_method import LIME
    return {
        'exstream': EXstream,
        'macrobase': MacroBase,
        'lime': LIME
    }
