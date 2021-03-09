import os
import math
import time

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from explanation.explainers import Explainer, get_list_entropy
from explanation.macrobase.helpers import classify_records, get_risk_ratio, get_categorical_features


class MacroBase(Explainer):
    """MacroBase explanation discovery method.

    See https://cs.stanford.edu/~deepakn/assets/papers/macrobase-sigmod17.pdf for details.
    """
    def __init__(self, args, output_path, explained_model=None, normal_model_samples=None):
        # this method never uses an explained model nor normal model samples
        super().__init__(args, output_path)
        # method-specific sample parameters
        self.feature_names = None
        # method-specific hyperparameters
        self.min_risk_ratio = args.macrobase_min_risk_ratio
        self.n_bins = args.macrobase_n_bins
        self.min_support = args.macrobase_min_support

    def fit_sample_parameters(self, sample, sample_labels):
        self.feature_names = range(sample.shape[1])

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
            # important features, FP-tree and normal bin boundaries
            important_fts, FPtree, boundaries = self.explain_records(normal_train, anomalous_train)
            runs_important_fts.append(important_fts)
            # classification performance using the explanations as rules
            y_true = np.concatenate([np.zeros(shape=len(normal_test)), np.ones(shape=len(anomalous_test))])
            y_pred = classify_records(np.concatenate(
                [normal_test, anomalous_test]), FPtree.iloc[0]['itemsets'], boundaries
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
        """Returns the important features, FP-Tree and normal bin boundaries for the provided
            normal and anomalous records.

        Args:
            normal_records (ndarray): normal records of shape `(n_normal_records, n_features)`.
            anomalous_records (ndarray): anomalous records of shape `(n_anomalous_records, n_features)`.

        Returns:
            list, pd.DataFrame, ndarray: important features, frequent pattern tree and normal bin boundaries.
        """
        # get attributes with sufficient relative risk and support for each features
        attributes, boundaries = self.records_to_attributes(normal_records, anomalous_records)

        # get discretized features
        boundaries = np.array(boundaries)
        cat_normal = get_categorical_features(normal_records, boundaries)
        cat_anomalous = get_categorical_features(anomalous_records, boundaries)

        transactions = pd.DataFrame(columns=attributes.keys())
        for v in cat_anomalous:
            row = {}
            for item in attributes.keys():
                if item in v:
                    row[item] = True
                else:
                    row[item] = False
            transactions = transactions.append(row, ignore_index=True)

        # get FP-tree
        FPtree = fpgrowth(transactions, min_support=self.min_support, use_colnames=True)

        # compute risk ratios
        risk_ratio, risk_size = [], []
        for item in FPtree['itemsets']:
            risk_ratio.append(get_risk_ratio(cat_normal, cat_anomalous, item))
            risk_size.append(len(item))
        FPtree['riskratio'] = risk_ratio
        FPtree['risksize'] = risk_size
        FPtree = FPtree.sort_values(by=['riskratio', 'support', 'risksize'], ascending=[False, False, False])

        # extract important features from the itemset
        important_fts = sorted([int(x.split('_')[0]) for x in list(FPtree.iloc[0]['itemsets'])])

        return important_fts, FPtree, boundaries

    def records_to_attributes(self, normal_records, anomalous_records):
        """Returns attributes with sufficient relative risk and support for each feature.

        Attributes are returned as a dictionary of the form `{attribute_name: attribute_support}`.
        Where an attribute name is a feature name + value range.

        Args:
            normal_records (ndarray): normal records of shape `(n_normal_records, n_features)`.
            anomalous_records (ndarray): anomalous records of shape `(n_anomalous_records, n_features)`.

        Returns:
            dict, list: the attributes for each feature, along with the value bin boundaries
                computed for each feature of `normal_records`.
        """
        attributes = {}
        normal_boundaries = []
        for ft in range(normal_records.shape[1]):
            ft_risky_attributes, ft_normal_boundaries = self.get_risky_attributes(
                anomalous_records[:, ft], normal_records[:, ft], self.feature_names[ft]
            )
            for attribute, support in ft_risky_attributes.items():
                attributes[attribute] = support
            normal_boundaries.append(list(ft_normal_boundaries))
        return attributes, normal_boundaries

    def get_risky_attributes(self, tsa, tsr, feature_name):
        """Returns the attributes with sufficient support and relative risk for `tsa` and `tsr`.

        The minimum risk ratio is determined by `self.min_risk_ratio`.

        Attributes are returned as a dictionary of the form `{attribute_name: attribute_support}`.
        Where an attribute name is a feature name + value range.

        Args:
            tsa (array-like): univariate anomalous time series.
            tsr (array-like): univariate reference time series.
            feature_name (str): name of the feature for the provided time series.

        Returns:
            dict, ndarray: attributes with sufficient relative risk and support, along with the
                value bin boundaries computed for `tsr`.
        """
        # get minimum support
        s = tsa.shape[0] // self.n_bins

        # compute time series histogram (note: it might be better to use tsa instead of tsr)
        normal_bars, normal_bins = np.histogram(tsr, bins=self.n_bins)

        # replace actual lower and upper bounds by -/+inf
        normal_bins[0], normal_bins[-1] = -math.inf, math.inf
        anomalous_bars, anomalous_bins = np.histogram(tsa, bins=normal_bins)

        # compute relative risk ratio for each attribute (record relative risk for each bin)
        a0 = anomalous_bars
        a0_plus_i = anomalous_bars + normal_bars
        b0 = len(tsa) - anomalous_bars
        b0_plus_i = len(tsr) + len(tsa) - anomalous_bars - normal_bars

        # if b0 or a0_plus_i are 0, set them to 1
        for i in range(len(b0)):
            if b0[i] == 0:
                b0[i] = 1
        for i in range(len(a0_plus_i)):
            if a0_plus_i[i] == 0:
                a0_plus_i[i] = 1

        relative_risk = (a0 / a0_plus_i) / (b0 / b0_plus_i)
        relative_risk = np.nan_to_num(relative_risk)

        # find attributes having at least the minimum support and relative risk
        attributes = {}
        for i in range(self.n_bins):
            if relative_risk[i] > self.min_risk_ratio and anomalous_bars[i] > s:
                attribute_name = str(feature_name)
                attribute_name += f'_{i}'
                attributes[attribute_name] = anomalous_bars[i]
        return attributes, normal_bins
