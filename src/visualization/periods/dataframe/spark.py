"""Spark-specific period DataFrame visualization classes.
"""
import os

import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from visualization.periods.dataframe.base import DataFrameViewer


class DriverViewer(DataFrameViewer):
    """Shows driver-related metrics.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'driver'

    def get_period_view(self, period_df):
        cols = [
            'driver_StreamingMetrics_streaming_totalReceivedRecords_value',
            'driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value',
        ]
        # {unit: [list of columns using this axis]} => Here 2 distinct axes with different units
        axes = {'Records': ['Total Received Records'], 'ms': ['Last Batch Scheduling Delay']}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class ExtendedDriverViewer(DataFrameViewer):
    """Shows an extended set of driver-related metrics (adding processing time).
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'driver_extended'

    def get_period_view(self, period_df):
        cols = [
            'driver_StreamingMetrics_streaming_totalProcessedRecords_value',
            'driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value',
        ]
        start_col = 'driver_StreamingMetrics_streaming_lastCompletedBatch_processingStartTime_value'
        end_col = 'driver_StreamingMetrics_streaming_lastCompletedBatch_processingEndTime_value'
        # unit: [list of columns using this axis] => Here 2 distinct axes with different units
        axes = {
            'processed records': ['Processed Records'],
            'scheduling delay (ms)': ['Scheduling Delay'],
            'processing time (ms)': ['Processing Time']
        }
        df_t = period_df[cols]
        processing_time_col = (period_df[end_col] - period_df[start_col]).apply(lambda x: 0 if x < 0 else x)
        df_t = df_t.assign(processing_time=processing_time_col)
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class InputRateViewer(DataFrameViewer):
    """Shows the input rate of the data received by the driver.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'input_rate'

    def get_period_view(self, period_df):
        axes = {'Records/sec': ['Received Input Rate']}
        df_t = pd.DataFrame(period_df['driver_StreamingMetrics_streaming_totalReceivedRecords_value'].diff())
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class ExecutorsCPUTimeCountViewer(DataFrameViewer):
    """Shows executors' CPU-related metrics.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'execs_cpu_time_counts'

    def get_period_view(self, period_df):
        execs_range = range(1, 6)
        cols = [f'{i}_executor_cpuTime_count' for i in execs_range]
        # here all columns share the same axis for enabling a better visual comparison
        axes = {'Count': [f'Executor {i} CPU Time Count' for i in execs_range]}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class ExecutorsCPUTimeViewer(DataFrameViewer):
    """Shows executors' CPU-related metrics.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'execs_cpu_time'

    def get_period_view(self, period_df):
        execs_range = range(1, 6)
        cols = [f'{i}_executor_cpuTime_count' for i in execs_range]
        # here all columns share the same axis for enabling a better visual comparison
        axes = {'Count': [f'Executor {i} CPU Time' for i in execs_range]}
        df_t = period_df[cols].diff(1)
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class ExecutorsMemoryViewer(DataFrameViewer):
    """Shows executors' memory-related metrics.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'execs_mem'

    def get_period_view(self, period_df):
        execs_range = range(1, 6)
        cols = [f'{i}_jvm_heap_usage_value' for i in execs_range]
        # here all cols shared the same axis for enabling a better visual comparison
        axes = {'%': [f'Executor {i} JVM Heap Usage' for i in execs_range]}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class OSCPUViewer(DataFrameViewer):
    """Shows Operating System's CPU-related metrics.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'os_cpu'

    def get_period_view(self, period_df):
        nodes_range = range(5, 9)
        cols = [f'node{i}_CPU_ALL_Idle%' for i in nodes_range]
        axes = {'%': [f'Node {i} All CPUs %Idle' for i in nodes_range]}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class OSAverageCPUViewer(DataFrameViewer):
    """Shows Operating System's average CPUs Idle% time across nodes.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'os_average_cpu'

    def get_period_view(self, period_df):
        nodes_range = range(5, 9)
        cols = [f'node{i}_CPU_ALL_Idle%' for i in nodes_range]
        axes = {'%': ['Average All CPUs %Idle']}
        df_t = pd.DataFrame(period_df[cols].mean(axis=1))
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class OSMemoryViewer(DataFrameViewer):
    """Shows Operating System's memory-related metrics.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'os_mem'

    def get_period_view(self, period_df):
        nodes_range = range(5, 9)
        cols = [f'node{i}_MEM_memfree' for i in nodes_range]
        axes = {'MB': [f'Node {i} Free Memory' for i in nodes_range]}
        df_t = period_df[cols]
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class OSNetworkReadViewer(DataFrameViewer):
    """Shows Operating System's network read operations-related metrics.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'os_net_read'

    def get_period_view(self, period_df):
        nodes_range = range(5, 9)
        df_t = pd.DataFrame()
        for n in nodes_range:
            cols = [
              f'node{n}_NET_em1-read-KB/s',
              f'node{n}_NET_em2-read-KB/s',
              f'node{n}_NET_em3-read-KB/s',
              f'node{n}_NET_em4-read-KB/s',
              f'node{n}_NET_ib0-read-KB/s',
              f'node{n}_NET_idrac-read-KB/s',
              f'node{n}_NET_lo-read-KB/s'
            ]
            df_t[f'node{n}_avg-read'] = period_df[cols].mean(axis=1)
        axes = {'kB/s': [f'Node {i} Avg Read' for i in nodes_range]}
        df_t.columns = sum(axes.values(), [])
        return df_t, axes


class OSNetworkWriteViewer(DataFrameViewer):
    """Shows Operating System's network write operations-related metrics.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        super().__init__(smoothing, cmap_name)
        self.name = 'os_net_write'

    def get_period_view(self, period_df):
        nodes_range = range(5, 9)
        df_t = pd.DataFrame()
        for n in nodes_range:
            cols = [
              f'node{n}_NET_em1-write-KB/s',
              f'node{n}_NET_em2-write-KB/s',
              f'node{n}_NET_em3-write-KB/s',
              f'node{n}_NET_em4-write-KB/s',
              f'node{n}_NET_ib0-write-KB/s',
              f'node{n}_NET_idrac-write-KB/s',
              f'node{n}_NET_lo-write-KB/s'
            ]
            df_t[f'node{n}_avg-write'] = period_df[cols].mean(axis=1)
        axes = {'kB/s': [f'Node {i} Avg Write' for i in nodes_range]}
        df_t.columns = sum(axes.values(), [])
        return df_t, axes
