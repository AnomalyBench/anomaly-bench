"""EXstream helpers module.
"""
import math
import operator


import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler


def classify_records(records, important_fts, anomalous_ft_segments):
    """Returns the binary classification of each of the provided `records`
        using `important_fts` and `anomalous_ft_segments` as rules.

    Args:
        records (ndarray): input records to classify of shape `(n_records, n_features)`.
        important_fts (list): list of features to consider for classification.
        anomalous_ft_segments: list of anomalous segments for each feature.

    Returns:
        ndarray: the binary array of classifications of shape `(n_records,)`.
    """
    return np.array(
        [classify_record(r, important_fts, anomalous_ft_segments) for r in records]
    )


def classify_record(record, important_fts, anomalous_ft_segments):
    """Returns the binary classification for `record` using `important_fts`
        and `anomalous_ft_segments` as rules.

    A record is classified as anomalous if all its important features are anomalous.
    A feature is defined as anomalous if its value belongs to one of its anomalous segments.

    Args:
        record (ndarray): record to classify of shape `(n_features,)`
        important_fts (list): list of features to consider for classification.
        anomalous_ft_segments: list of anomalous segments for each feature.

    Returns:
        int: the binary classification for the record.
    """
    for ft, ft_segments in zip(important_fts, anomalous_ft_segments):
        ft_pred = 0
        for x1, x2 in ft_segments:
            # a feature is anomalous if its value belongs to an anomalous segment
            if x1 <= record[ft] <= x2:
                ft_pred = 1
                break
        # if not all features are anomalous, then the record is classified as normal
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
    """Returns the "false positive features" (i.e. to not consider) for the provided
        `normal_records`.

    Args:
        normal_records (ndarray): normal records of shape `(n_records, n_features)`.

    Returns:
        list: false positive features found for the provided records.
    """
    # remove features with unusually high standard deviation
    std = np.std(normal_records, axis=0)
    scaler = StandardScaler()
    D = scaler.fit_transform(std.reshape(-1, 1))
    false_positives = list(np.argwhere(D > 1.64)[:, 0])

    # remove features which are strictly increasing or decreasing
    for f in range(normal_records.shape[1]):
        if f in false_positives:
            continue
        else:
            # get moving mean
            window_size = normal_records.shape[0] // 5
            if window_size == 0:
                window_size = 1
            x = normal_records[:, f]
            avg_sliding_window = []
            for i in range(0, normal_records.shape[0], window_size):
                avg_window = np.mean(x[i:i + window_size])
                avg_sliding_window.append(avg_window)

            # check for increasing and decreasing tendencies
            increasing = True
            decreasing = True
            for i in range(1, len(avg_sliding_window)):
                if avg_sliding_window[i-1] > avg_sliding_window[i]:
                    increasing = False
                    break
            for i in range(1, len(avg_sliding_window)):
                if avg_sliding_window[i-1] < avg_sliding_window[i]:
                    decreasing = False
                    break
            if increasing or decreasing:
                false_positives.append(f)
    return false_positives


def entropy(x):
    """Returns the entropy of `x`."""
    x = np.asarray(x)
    return np.sum(-x * np.log(x))


def class_entropy(tsa, tsr):
    """Returns the class entropy of `tsa` and `tsr`.

    Args:
        tsa (array-like): univariate anomalous time series.
        tsr (array-like): univariate reference time series.

    Returns:
        float: the class entropy (h) of `tsa` and `tsr`.
    """
    l_a, l_r = len(tsa), len(tsr)
    s = l_a + l_r
    p_a, p_r = l_a / s, l_r / s
    return entropy([p_a, p_r])


def segmentation(tsa, tsr):
    """Returns normal, abnormal and mixed value segments based on the provided anomalous
        and reference time series.

    Args:
        tsa (array-like): univariate anomalous time series.
        tsr (array-like): univariate reference time series.

    Returns:
        list, list, list, int: the normal, abnormal and mixed segments (all as lists of lists). The last
            element returned is `len_segment == len(tsa) + len(tsr)`, only returned here for efficiency.
    """
    common_values = set(tsa).intersection(set(tsr))
    tagged_tsa = [(v, -1) for v in tsa if v not in common_values]
    tagged_tsr = [(v, 1) for v in tsr if v not in common_values]
    tagged_common_tsa = [(v, 0) for v in tsa if v in common_values]
    tagged_common_tsr = [(v, 0) for v in tsr if v in common_values]
    segment = tagged_tsa + tagged_tsr + tagged_common_tsa + tagged_common_tsr
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
    """Returns the reguralized segmentation entropy between two intervals
        given the output of the segmentation function.

    Returns:
        float: the reguralized segmentation entropy (H).
    """
    if len(normal) == 0 and len(abnormal) == 0:
        return 0.0
    else:
        H = entropy([x[2] / len_segment for x in normal])
        H += entropy([x[2] / len_segment for x in abnormal])
        H += entropy([x[2] / len_segment for x in mixed])
        # add the regularization term
        H += sum([x[2] for x in mixed]) * (1 / len_segment) * np.log(len_segment)
    return H


# single-feature rewards
def segmentation_entropy_reward(tsa, tsr, normal, abnormal, mixed, len_segment):
    """Returns the segmentation entropy reward for `tsa` and `tsr`
        given the output of the segmentation function.

    Returns:
        float: the segmentation entropy reward for `tsa` and `tsr`.
    """
    H_class = class_entropy(tsr, tsa)
    H_segmentation = segmentation_entropy(normal, abnormal, mixed, len_segment)
    if H_segmentation == 0.0:
        return 0.0
    else:
        return H_class / H_segmentation
