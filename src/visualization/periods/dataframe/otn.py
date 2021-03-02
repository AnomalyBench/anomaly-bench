"""OTN-specific period DataFrame visualization classes.

/!\\ `Output` corresponds to a service's input node, `Input` to a service's output node.
"""
import os
from abc import abstractmethod

import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from visualization.periods.dataframe.base import DataFrameViewer
from visualization.helpers.otn import get_service_locations


class ServiceDataFramesViewer(DataFrameViewer):
    """Service DataFrames visualization class.

    Provides an additional method for plotting in the same figure the chronologically
    ranked periods of the provided service.

    Args:
        smoothing (int): records of the series are smoothed using a rolling average of `smoothing` seconds.
        cmap_name (str): name of the color map used for plotting the selected columns.
        zones_info (list): optional list of
            `[primary_start, secondary_start, secondary_end, tolerance_end]` used
            for highlighting the 3 anomaly zones of a "fast alert" evaluation.
    """
    def __init__(self, smoothing=60, cmap_name='Set2', zones_info=None):
        super().__init__(smoothing, cmap_name)
        self.zones_info = zones_info

    @abstractmethod
    def get_period_view(self, period_df):
        """Repeats the method definition to enable defining the class.
        """

    def plot_service_periods(self, service_no, datasets_df, datasets_info):
        """Plots all the periods of the provided service within the same figure.

        In the title will be shown the datasets in which the periods belong to.

        Args:
            service_no (int): service number (i.e. name) to show (the same as its file name).
            datasets_df (dict): datasets of the form `{set_name: period_dfs}`.
            datasets_info (dict): datasets information of the form {`set_name`: `periods_info`};
                with `periods_info` a list of the form `[file_name, trace_type, period_rank]`
                for each period of the set.
        """
        # chronologically ranked list of periods `(set_name, set_position)` for the service
        service_locations = get_service_locations(service_no, datasets_info)
        period_dfs, periods_info, service_sets = [], [], []
        for (service_set, service_idx) in service_locations:
            period_dfs.append(datasets_df[service_set][service_idx])
            periods_info.append(datasets_info[service_set][service_idx])
            service_sets.append(service_set.capitalize())
        self.plot_periods(
            period_dfs, periods_info, f'Service #{service_no} ({", ".join(service_sets)})',
            zones_info=self.zones_info
        )


class GlobalViewer(ServiceDataFramesViewer):
    """Shows the log-power of the input and output data and supervision signals.
    """
    def __init__(self, smoothing=60, cmap_name='Set2', zones_info=None):
        super().__init__(smoothing, cmap_name, zones_info)
        self.name = 'global'

    def get_period_view(self, period_df):
        cols = [
            'TotalOutputPower',
            'TotalInputPower',
            'LaserOutputPower',
            'LaserInputPower'
        ]
        # use a dedicated axis for each metric, despite them being in the same unit
        axes = {
            'Log-Power #1': ['Input Data Signal'],
            'Log-Power #2': ['Output Data Signal'],
            'Log-Power #3': ['Input Supervision Signal'],
            'Log-Power #4': ['Output Supervision Signal']
        }
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class GainsViewer(ServiceDataFramesViewer):
    """Shows network gain metrics for the data and supervision signals.
    """
    def __init__(self, smoothing=60, cmap_name='Set2', zones_info=None):
        super().__init__(smoothing, cmap_name, zones_info)
        self.name = 'gains'

    def get_period_view(self, period_df):
        df_t, axes = pd.DataFrame(), dict()
        # signal label and signal text
        for sl, st in zip(['Total', 'Laser'], ['Data', 'Supervision']):
            df_t[f'{sl}Gain'] = period_df[f'{sl}InputPower'] - period_df[f'{sl}OutputPower']
            axes[f'{st} Gain'] = [f'{st} Signal Gain']
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class DataSignalsViewer(ServiceDataFramesViewer):
    """Shows the log-power of the input and output data signals.
    """
    def __init__(self, smoothing=60, cmap_name='Set2', zones_info=None):
        super().__init__(smoothing, cmap_name, zones_info)
        self.name = 'data_signal'

    def get_period_view(self, period_df):
        cols = ['TotalOutputPower', 'TotalInputPower']
        # use a dedicated axis for each metric, despite them being in the same unit
        axes = {'Log-Power #1': ['Input Data Signal'], 'Log-Power #2': ['Output Data Signal']}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class SupervisionSignalsViewer(ServiceDataFramesViewer):
    """Shows the log-power of the input and output supervision signals.
    """
    def __init__(self, smoothing=60, cmap_name='Set2', zones_info=None):
        super().__init__(smoothing, cmap_name, zones_info)
        self.name = 'supervision_signal'

    def get_period_view(self, period_df):
        cols = ['LaserOutputPower', 'LaserInputPower']
        # use a dedicated axis for each metric, despite them being in the same unit
        axes = {'Log-Power #1': ['Input Supervision Signal'], 'Log-Power #2': ['Output Supervision Signal']}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class InputSignalsViewer(ServiceDataFramesViewer):
    """Shows the log-power of the input data and supervision signals.
    """
    def __init__(self, smoothing=60, cmap_name='Set2', zones_info=None):
        super().__init__(smoothing, cmap_name, zones_info)
        self.name = 'input_signals'

    def get_period_view(self, period_df):
        cols = ['TotalOutputPower', 'LaserOutputPower']
        # use a dedicated axis for each metric, despite them being in the same unit
        axes = {'Log-Power #1': ['Input Data Signal'], 'Log-Power #2': ['Input Supervision Signal']}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class OutputSignalsViewer(ServiceDataFramesViewer):
    """Shows the log-power of the output data and supervision signals.
    """
    def __init__(self, smoothing=60, cmap_name='Set2', zones_info=None):
        super().__init__(smoothing, cmap_name, zones_info)
        self.name = 'output_signals'

    def get_period_view(self, period_df):
        cols = ['TotalInputPower', 'LaserInputPower']
        # use a dedicated axis for each metric, despite them being in the same unit
        axes = {'Log-Power #1': ['Output Data Signal'], 'Log-Power #2': ['Output Supervision Signal']}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes
