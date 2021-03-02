Exathlon
==============================

> **A Benchmark for Explainable Anomaly Detection over Time Series**
>
> **Abstract:** *Access to high-quality data repositories and benchmarks have been instrumental in advancing the state of the art in many experimental research domains. While advanced analytics tasks over time series data have been gaining lots of attention, lack of such community resources severely limits scientific progress. In this paper, we present Exathlon, the first comprehensive public benchmark for explainable anomaly detection over high-dimensional time series data. Exathlon has been systematically constructed based on real data traces from repeated executions of large-scale stream processing jobs on an Apache Spark cluster. Some of these executions were intentionally disturbed by introducing instances of six different types of anomalous events (e.g., misbehaving inputs, resource contention, process failures). For each of the anomaly instances, ground truth labels for the root cause interval as well as those for the extended effect interval are provided, supporting the development and evaluation of a wide range of anomaly detection (AD) and explanation discovery (ED) tasks.We demonstrate the practical utility of Exathlon’s dataset, evaluation methodology, and end-to end data science pipeline design through an experimental study with three state-of-the-art AD and ED techniques.*

This repository contains the source code for the experimental study of the paper.

## Configure this Project

Using `conda`, from the project root folder, execute the following commands: 

```bash
conda create -n exathlon python=3.7
conda activate exathlon
conda install -c conda-forge --yes --file requirements.txt
```

At the root of the project folder, create a `.env` file containing the lines:

```bash
DATA_ROOT=path/to/used/data/root
OUTPUTS_ROOT=path/to/pipeline/outputs
```

The pipeline outputs refer to all the outputs that will be produced by the AD pipeline, including intermediate and fully processed data, models, model information and final results.

### Note: Running this Project on Windows

Some results and logging paths might exceed the Windows historical path length limitation of 260 characters, leading to some errors when running the pipeline. To counter this, we advise to disable this limitation following the procedure described in the [official Python documentation](https://docs.python.org/3/using/windows.html):

>Windows historically has limited path lengths to 260 characters. This meant that paths longer than this would not resolve and errors would result.
>
>In the latest versions of Windows, this limitation can be expanded to approximately 32,000 characters. Your administrator will need to activate the “Enable Win32 long paths” group policy, or set `LongPathsEnabled` to `1` in the registry key `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`.
>
>This allows the [`open()`](https://docs.python.org/3/library/functions.html#open) function, the [`os`](https://docs.python.org/3/library/os.html#module-os) module and most other path functionality to accept and return paths longer than 260 characters.
>
>After changing the above option, no further configuration is required.
>
>Changed in version 3.6: Support for long paths was enabled in Python. 