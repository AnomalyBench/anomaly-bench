import numpy as np


def classify_records(records, itemset, boundaries=None):
    classified = records.copy() if boundaries is None else get_categorical_features(records, boundaries)
    return np.array([classify_record(r, itemset) for r in classified])


def classify_record(record, itemset):
    for item in itemset:
        if item not in record:
            return 0
    return 1


def get_categorical_features(records, boundaries):
    data = records.transpose()
    feature_count, row_count = data.shape
    result = [[''] * row_count for i in range(feature_count)]
    for i in range(feature_count):
        for j in range(row_count):
            result[i][j] = f'{i}_{get_category(data[i, j], boundaries[i])}'
    return np.array(result).transpose()


def get_category(value, boundaries):
    for i in range(len(boundaries)):
        if boundaries[i] <= value < boundaries[i+1]:
            return i


def risk_ratio_cal(normal_window, anomalous_window, itemset):
    normal_window_label = [classify_record(v, itemset) for v in normal_window]
    anomalous_window_label = [classify_record(v, itemset) for v in anomalous_window]
    a0 = sum(x == 1 for x in anomalous_window_label)
    ai = sum(x == 1 for x in normal_window_label)
    b0 = sum(x == 0 for x in anomalous_window_label)
    bi = sum(x == 0 for x in normal_window_label)
    if b0 == 0:
        b0 = 1
    if a0 == 0:
        risk = 0
    else:
        risk = (a0/(a0+ai))/(b0/(b0+bi))
    return risk
