"""Module gathering various explanation discovery methods to compare.
"""
import os
from abc import abstractmethod

import numpy as np
from scipy.stats import entropy


def get_list_entropy(values_list, upper_bound):
    """Returns the entropy of `values_list` given `upper_bound`.

    In our context, `upper_bound` will typically be the number of features.

    Args:
        values_list (list): values to compute the entropy of.
        upper_bound (int): upper-bound for the computed entropy.

    Returns:
        float: the entropy of `values_list` given `upper_bound`.
    """
    freq = []
    for i in range(upper_bound):
        if values_list.count(i) != 0:
            freq.append(values_list.count(i)/len(values_list))
    return entropy(freq, base=2)


class Explainer:
    """Explainer model base class.

    All explanation discovery methods we consider for now operate on `(normal + anomaly)` samples.

    Each method is fit per sample and the fit parameters are used for explaining the sample only.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the scoring model and information to.
        explained_model (Scorer|None): outlier score assignment model to explain if relevant.
        normal_model_samples (ndarray|None): normal samples in the scoring model format, of shape
            `(n_samples, sample_size, n_features)`. Such samples are directly assigned outlier scores
            by the scorer.
    """
    def __init__(self, args, output_path, explained_model=None, normal_model_samples=None):
        self.output_path = output_path
        self.explained_model = explained_model
        self.normal_model_samples = normal_model_samples
        # parameters fit to each sample
        self.normal_part, self.anomalous_part = None, None

    def fit_sample(self, sample, sample_labels):
        """Fits parameters to the provided samples.

        Args:
            sample (ndarray): sample of shape `(sample_size, n_features)`, consisting in a
                normal period followed by an anomalous period.
            sample_labels (ndarray): multiclass labels for the sample, of shape `(sample_size,)`.
        """
        # setup references to the normal and anomalous parts of the sample
        anomaly_start, anomaly_end = np.where(sample_labels > 0)[0][0], len(sample_labels)
        self.normal_part, self.anomalous_part = sample[:anomaly_start], sample[anomaly_start:]
        # call any method-specific sample fitting of the child class
        self.fit_sample_parameters(sample, sample_labels)

    def fit_evaluate_samples(self, samples, samples_labels, test_prop=0.2, n_runs=5):
        """Fits and evaluates the explanation discovery method on `samples` and `samples_labels`.

        Args:
            samples (ndarray): samples of shape `(n_samples, sample_size, n_features)`, with `sample_size`
                depending on the sample.
            samples_labels (ndarray): multiclass labels for each sample of shape `(n_samples, sample_size,)`.
            test_prop (float): proportion of each sample used as a test set when reporting classification
                metrics (and possibly constituting different runs).
            n_runs (int): number of explanation discovery runs when reporting stability.

        Returns:
            dict: the evaluated metrics on the samples as `{metric_name: metric_value}`.
        """
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
        metrics_dict['entropy'] = get_list_entropy(important_fts, n_features)
        # average the rest of the metrics across samples
        for m_name in [k for k in metrics_dict if k not in ['consistency', 'entropy']]:
            metrics_dict[m_name] = sum(metrics_dict[m_name]) / len(metrics_dict[m_name])
        # reorder and return the explanation metrics
        returned_metrics = dict()
        shared_metrics = ['fit_eval_time', 'local_stability', 'consistency', 'entropy']
        classification_metrics = ['f1-score', 'accuracy', 'precision', 'recall']
        for m_name in shared_metrics + classification_metrics:
            # if classification metrics are not returned, fill them with NaNs
            if m_name in classification_metrics and m_name not in metrics_dict:
                returned_metrics[m_name] = np.nan
            # all methods must return the shared metrics
            else:
                returned_metrics[m_name] = metrics_dict[m_name]
        return metrics_dict

    @abstractmethod
    def fit_sample_parameters(self, sample, sample_labels):
        """Fits any additional parameters for the sample before evaluating its explanation(s).
        """

    @abstractmethod
    def fit_evaluate_sample(self, sample, sample_labels, test_prop=0.2, n_runs=5):
        """Fits and evaluates the explanation discovery method on `sample` and `sample_labels`.
        """


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
