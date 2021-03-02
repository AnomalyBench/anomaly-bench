"""Spark-specific visualization helpers.
"""
import os

import matplotlib.pyplot as plt

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(src_path)
from utils.spark import EVENT_TYPES

# colors used for representing each type of event
dark2_palette = plt.get_cmap('Dark2').colors
ANOMALY_COLORS = {type_: dark2_palette[i] for i, type_ in enumerate(EVENT_TYPES)}

# colors used for reporting performance globally and per event type
METRICS_COLORS = dict({'global': 'blue'}, **ANOMALY_COLORS)
