"""LIME helpers module.
"""
import re


def get_important_fts(exp):
    """Returns the "important" features from the LIME explanation `exp`.

    Args:
        exp (Explanation): explanation object produced by LIME.

    Returns:
        list: sorted list of important feature indices (sorted by feature index).
    """
    important_fts = []
    for e in exp.as_list():
        feature = re.findall(r'ft_\d+', e[0])
        v = int(feature[0][3:])
        important_fts.append(v)
    return sorted(list(set(important_fts)))
