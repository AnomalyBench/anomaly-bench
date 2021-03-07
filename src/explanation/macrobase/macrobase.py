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
from explanation.explainers import Explainer, get_entropy
from explanation.macrobase.helpers import classify_records, risk_ratio_cal, get_categorical_features


class MacroBase(Explainer):
    """MacroBase explanation discovery method.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path)
        # method-specific sample parameters
        self.feature_names = None
        # method-specific hyperparameters
        self.r = args.macrobase_r
        self.n_bins = args.macrobase_n_bins
        self.min_support = args.macrobase_min_support

    def fit_sample_parameters(self, sample, sample_labels):
        self.feature_names = range(sample.shape[1])

    def fit_evaluate_sample(self, sample, sample_labels, test_prop=0.2, n_runs=5):
        metrics = dict()
        # fit sample parts, feature clusters and false positive features
        self.fit_sample(sample, sample_labels)

        # get important features using the whole sample data
        important_fts, exp, boundaries = self.explain_records(self.normal_part, self.anomalous_part)
        metrics['important_fts'] = important_fts

        # get local stability and average classification performance using `n_runs` random splits
        start_time = time.time()
        runs_important_fts, classification_keys = [], ['accuracy', 'precision', 'recall', 'f1-score']
        for k in classification_keys:
            metrics[k] = []
        for _ in range(n_runs):
            # generate train test split
            normal_train, normal_test, = train_test_split(self.normal_part, test_size=test_prop, random_state=None)
            anomalous_train, anomalous_test, = train_test_split(
                self.anomalous_part, test_size=test_prop, random_state=None
            )
            important_fts, exp, boundaries = self.explain_records(normal_train, anomalous_train)
            runs_important_fts.append(important_fts)
            # classification performance using the explanation rules
            y_true = np.concatenate([np.zeros(shape=(len(normal_test),)), np.ones(shape=(len(anomalous_test),))])
            y_pred = classify_records(np.concatenate(
                [normal_test, anomalous_test]), exp.iloc[0]['itemsets'], boundaries
            )
            metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            for k, v in zip(classification_keys[1:], [p, r, f]):
                metrics[k].append(v)
        # record average time to fit and evaluation the explanation method
        metrics['fit_eval_time'] = (time.time() - start_time) / n_runs
        # average classification metrics across runs
        for k in classification_keys:
            metrics[k] = sum(metrics[k]) / len(metrics[k])
        # use important features across runs to compute the local stability
        metrics['local_stability'] = get_entropy(sample.shape[1], runs_important_fts)
        return metrics

    def explain_records(self, normal, anomalous):
        # getting the discrete attributes
        attributes, boundary = self.trace_to_attributes(normal, anomalous)

        boundary = np.array(boundary)
        cat_normal = get_categorical_features(normal, boundary)
        cat_anomalous = get_categorical_features(anomalous, boundary)

        transactions = pd.DataFrame(columns=attributes.keys())
        for v in cat_anomalous:
            row = {}
            for item in attributes.keys():
                if item in v:
                    row[item] = True
                else:
                    row[item] = False
            transactions = transactions.append(row, ignore_index=True)

        # Mining FP-tree
        FPtree = fpgrowth(transactions, min_support=self.min_support, use_colnames=True)

        # removing patterns with insufficient relative risk ratio
        risk_ratio_cal(cat_normal, cat_anomalous, FPtree.iloc[0]['itemsets'])

        risk_ratio = []
        risk_size = []
        for item in FPtree['itemsets']:
            risk_ratio.append(risk_ratio_cal(cat_normal, cat_anomalous, item))
            risk_size.append(len(item))

        FPtree['riskratio'] = risk_ratio
        FPtree['risksize'] = risk_size

        FPtree = FPtree.sort_values(by=['riskratio', 'support', 'risksize'], ascending=[False, False, False])

        important_fts = sorted([int(x.split('_')[0]) for x in list(FPtree.iloc[0]['itemsets'])])

        return important_fts, FPtree, boundary

    def trace_to_attributes(self, normal_values, anomalous_values):
        trace_attributes = {}
        boundary = []
        for f in range(normal_values.shape[1]):
            window_attributes, boundary_item = self.get_risky_attributes(
                normal_values[:, f], anomalous_values[:, f], self.feature_names[f]
            )
            for attribute, support in window_attributes.items():
                trace_attributes[attribute] = support
            boundary.append(list(boundary_item))
        return trace_attributes, boundary

    def get_risky_attributes(self, normal_window, anomalous_window, feature_name):
        # minimum support
        s = anomalous_window.shape[0] // self.n_bins

        # computing time-series histogram; also it might be better to use ano in place of normal
        normal_bars, normal_bins = np.histogram(normal_window, bins=self.n_bins)

        # indeed won't use the lower and upper, replace by inf
        normal_bins[0] = -math.inf
        normal_bins[-1] = math.inf
        anomalous_bars, anomalous_bins = np.histogram(anomalous_window, bins=normal_bins)

        # computing relative risk ratio for each attribute
        # record relative_risk for each bin
        a0 = anomalous_bars
        a0plusi = anomalous_bars + normal_bars
        b0 = len(anomalous_window) - anomalous_bars
        b0plusi = len(normal_window) + len(anomalous_window) - anomalous_bars - normal_bars

        # special case, if b0 == 0; set it to 1
        for i in range(len(b0)):
            if b0[i] == 0:
                b0[i] = 1

        # special case, if a0plusi == 0;
        for i in range(len(a0plusi)):
            if a0plusi[i] == 0:
                a0plusi[i] = 1

        relative_risk = (a0 / a0plusi) / (b0 / b0plusi)
        relative_risk = np.nan_to_num(relative_risk)

        # finding attributes with minimum support and relative risk
        # the attributes is a dict.
        # The key is the string including the feature name and the range;
        # the value is the count or support in this range
        attributes = {}
        for i in range(self.n_bins):
            if relative_risk[i] > self.r and anomalous_bars[i] > s:
                attribute_name = str(feature_name)
                attribute_name += "_" + str(i)
                attributes[attribute_name] = anomalous_bars[i]
        return attributes, normal_bins
