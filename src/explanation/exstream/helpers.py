import operator
import math

import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler


def classify_records(sample, important_fts, anomalous_ft_intervals):
    return np.array(
        [classify_record(r, important_fts, anomalous_ft_intervals) for r in sample]
    )


def classify_record(record, important_fts, anomalous_ft_intervals):
    """Form (A OR B OR ...) AND (C OR D OR ...) AND ..."""
    for ft, ft_intervals in zip(important_fts, anomalous_ft_intervals):
        ft_pred = 0
        for x1, x2 in ft_intervals:
            if x1 <= record[ft] <= x2:
                ft_pred = 1
                break
        if ft_pred == 0:
            return 0
    return 1


def get_feature_clusters(sample_df):
    """Returns the clusters for the provided sample features.

    Args:
        sample_df (pd.DataFrame): sample DataFrame whose features to cluster.

    Returns:
        ndarray: the flat cluster to which each feature belongs.
    """
    correlation_matrix = sample_df.corr()
    correlation_matrix = np.abs(np.nan_to_num(correlation_matrix.values))
    pdist = sch.distance.pdist(correlation_matrix)
    linkage = sch.linkage(pdist, method='complete')
    return sch.fcluster(linkage, 0.5 * pdist.max(), 'distance')


def get_fp_features(normal_records):
    # removing features with unusually high standard deviation
    std = np.std(normal_records, axis=0)
    scaler = StandardScaler()
    D = scaler.fit_transform(std.reshape(-1, 1))
    false_positives = list(np.argwhere(D > 1.64)[:, 0])

    # removing features which are striclty increasing or decreasing
    for f in range(normal_records.shape[1]):
        if f in false_positives:
            continue
        else:
            # getting the moving mean
            window_size = normal_records.shape[0] // 5
            if window_size == 0:
                window_size = 1
            x = normal_records[:, f]
            avg_sliding_window = []
            for i in range(0, normal_records.shape[0], window_size):
                avg_window = np.mean(x[i:i + window_size])
                avg_sliding_window.append(avg_window)

            increasing = True
            decreasing = True
            # checking fro increasing tendency
            for i in range(1, len(avg_sliding_window)):
                if avg_sliding_window[i - 1] > avg_sliding_window[i]:
                    increasing = False
                    break
            for i in range(1, len(avg_sliding_window)):
                if avg_sliding_window[i - 1] < avg_sliding_window[i]:
                    decreasing = False
                    break

            if increasing or decreasing:
                false_positives.append(f)
    return false_positives


def entropy(x):
    x = np.asarray(x)
    return np.sum(-x * np.log(x))


def class_entropy(TSA, TSR):
    """
    :param TSA: an abnormal trace time series (list)
    :param TSR: a normal reference trace time series (list)
    returns:
    h: the class entropy of the two time series
    """
    l_A = len(TSA)
    l_R = len(TSR)
    s = l_A + l_R
    p_A = l_A / s
    p_R = l_R / s
    h = entropy([p_A, p_R])
    return h


def segmentation(TSA, TSR):
    """
    :param TSA: an abnormal trace time series (list)
    :param TSR: a normal reference trace time series (list)
    return:
    normal: the normal segments  (list of list)
    abnormal: the abnormal segments (list of list)
    mixed: the mixed segments (list of list)
    len_segment: len(TSA)+len(TSR)
    """
    common_values = set(TSA).intersection(set(TSR))
    tagged_TSA = [(v, -1) for v in TSA if v not in common_values]
    tagged_TSR = [(v, 1) for v in TSR if v not in common_values]
    tagged_common_TSA = [(v, 0) for v in TSA if v in common_values]
    tagged_common_TSR = [(v, 0) for v in TSR if v in common_values]
    segment = tagged_TSA + tagged_TSR + tagged_common_TSA + tagged_common_TSR
    segment.sort(key=operator.itemgetter(0))
    segment[0] = (-math.inf, segment[0][1])
    segment[-1] = (math.inf, segment[-1][1])
    normal = []
    abnormal = []
    mixed = []

    i = 0
    len_segment = len(segment)
    while i < len_segment:
        seg = []
        if segment[i][1] == -1:
            while i < len_segment and segment[i][1] == -1:
                seg.append(segment[i][0])
                i += 1
            abnormal.append((min(seg), max(seg), len(seg)))
        elif segment[i][1] == 0:
            while i < len_segment and segment[i][1] == 0:
                seg.append(segment[i][0])
                i += 1
            mixed.append((min(seg), max(seg), len(seg)))
        else:
            while i < len_segment and segment[i][1] == 1:
                seg.append(segment[i][0])
                i += 1
            normal.append((min(seg), max(seg), len(seg)))
    return normal, abnormal, mixed, len_segment


def segmentation_entropy(normal, abnormal, mixed, len_segment):
    """
    computes the reguralized segmentation entropy between two intervals

    input: the result of the segmentation function

    return:
    H: reguralized segmentation entropy (float)
    """
    if len(normal) == 0 and len(abnormal) == 0:
        return 0.0
    else:

        H = entropy([x[2] / len_segment for x in normal])
        H += entropy([x[2] / len_segment for x in abnormal])
        H += entropy([x[2] / len_segment for x in mixed])
        H += sum([x[2] for x in mixed]) * (1 / len_segment) * np.log(len_segment)  # regularization term
    return H


# single-feature rewards
def segmentation_entropy_reward(normal_trace, abnormal_trace, normal, abnormal, mixed, len_segment):
    """
    Computes the segmentation entropy reward for a normal trace and an abnormal trace
    :param: normal_trace: 1d np.array the normal time series
    :param: abnormal_trace: 1d np.array the abnormal time series
    :return:
    reward: the segmentation entropy reward between the input traces
    """
    H_class = class_entropy(normal_trace, abnormal_trace)
    H_segmentation = segmentation_entropy(normal, abnormal, mixed, len_segment)
    if H_segmentation == 0.0:
        return 0.0
    else:
        return H_class / H_segmentation


def predict_record(record, important_fs, anomalous_segments):
    """check whether a instance is anomalous."""
    flag = 1
    for f_index, f_segs in zip(important_fs, anomalous_segments):
        flag = flag * partial_exp(record, f_index, f_segs)
    return flag


def partial_exp(record, feature, segs):
    """check whether a instance satisfy a predicates in disconjunction."""
    flag = 0
    for seg in segs:
        if seg[0] <= record[feature] <= seg[1]:
            flag = 1
            break
    return flag
