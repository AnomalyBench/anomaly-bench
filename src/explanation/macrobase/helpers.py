"""MacroBase helpers module.
"""
import numpy as np


def classify_records(records, itemset, boundaries=None):
    """Returns the binary classifications for `records` using the conjunction of the
        predicates in `itemset` as a rule.

    If `boundaries` is None, records are assumed categorical, hence they will be categorized using
    `boundaries`.

    Args:
        records (ndarray): input records to classify of shape `(n_records, n_features)`.
        itemset (frozenset): categorical predicates whose conjunctive satisfaction makes a record anomalous.
        boundaries (array-like|None): optional boundaries used to discretize input records.

    Returns:
        ndarray: the binary array of classifications of shape `(n_records,)`.
    """
    classified = records if boundaries is None else get_categorical_features(records, boundaries)
    return np.array([classify_record(r, itemset) for r in classified], dtype=int)


def classify_record(cat_record, itemset):
    """Returns the binary classifications for `cat_record` using the conjunction of the
        predicates in `itemset` as a rule.

    Args:
        cat_record (ndarray): input categorical record to classify of shape `(n_features,)`.
        itemset (frozenset): categorical predicates whose conjunctive satisfaction makes a record anomalous.

    Returns:
        int: the binary classification for the record.
    """
    # return 1 if the record satisfies the itemset, 0 otherwise
    for item in itemset:
        if item not in cat_record:
            return 0
    return 1


def get_categorical_features(records, boundaries):
    """Returns the record features discretized using `boundaries`.

    The returned value for a feature will be a string containing its index in the feature list followed by
        the index of the boundary bin it belongs to.

    Args:
        records (ndarray): records whose features to categorize of shape `(n_records, n_features)`.
        boundaries (ndarray): value bin boundaries for each feature, of shape `(n_features, n_boundaries)`.

    Returns:
        ndarray: the input records with discretized features of shape `(n_records, n_features)`.
    """
    # transpose records to get the feature as the first axis
    data = records.transpose()
    feature_count, row_count = data.shape
    result = [[''] * row_count for _ in range(feature_count)]
    for i in range(feature_count):
        for j in range(row_count):
            result[i][j] = f'{i}_{get_boundary_idx(data[i, j], boundaries[i])}'
    return np.array(result).transpose()


def get_boundary_idx(value, boundaries):
    """Returns the boundary index of the provided value.
    """
    for i in range(len(boundaries)):
        if boundaries[i] <= value < boundaries[i+1]:
            return i


def get_risk_ratio(cat_normal_records, cat_anomalous_records, itemset):
    """Returns the risk ratio for the provided normal, anomalous records and itemset.

    Args:
        cat_normal_records (ndarray): normal records with discretized features
            of shape `(n_normal_records, n_features)`.
        cat_anomalous_records (ndarray): anomalous records with discretized features
            of shape `(n_normal_records, n_features)`.
        itemset (frozenset): categorical predicates whose conjunctive satisfaction makes a record anomalous.

    Returns:
        float: the risk ratio.
    """
    # get binary classifications for the input records
    normal_preds = [classify_record(v, itemset) for v in cat_normal_records]
    anomalous_preds = [classify_record(v, itemset) for v in cat_anomalous_records]
    a0 = sum(x == 1 for x in anomalous_preds)
    ai = sum(x == 1 for x in normal_preds)
    b0 = sum(x == 0 for x in anomalous_preds)
    bi = sum(x == 0 for x in normal_preds)
    if b0 == 0:
        b0 = 1
    if a0 == 0:
        risk = 0
    else:
        risk = (a0 / (a0 + ai)) / (b0 / (b0 + bi))
    return risk
