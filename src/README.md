Source Folder
==============================

This folder contains all the source code for the full Anomaly Detection pipeline.

## Packages Description

**`utils` - General-purpose utility package, mainly handling paths and command-line arguments.**

- `utils.common` - Utility elements common to all datasets supported by the pipeline.
- `utils.spark` - Spark Streaming-specific utility elements and meta-information.
- `utils.otn` - Optical Transport Network-specific utility elements and meta-information.

**`metrics` - Precision and Recall definition package.**

Depending on the dataset, labels and scenario, we might want to use different definitions of Precision and Recall when evaluating the binary predictions produced by the AD pipeline.

**`data` - Step #1: constituting the pipeline's `train/threshold/test` sets.**

The outputs of this step are 3 lists of contiguous period `DataFrames`. Each period can be of any length and has an additional `Anomaly` column specifying whether a given record belongs to an anomaly or not. The sets are saved as pickle files under `OUTPUTS_ROOT/data/interim` along with sets 'information' holding any additional information we want to store about the sets periods. 

* The `train` set will be used to model the data's normal behavior, and should hence be a list of normal/undisturbed periods.
* The `threshold` set will be used to calibrate the outlier score threshold, and should hence be either a list of normal/undisturbed periods or a list of periods containing both normal and anomalous data. Only the latter is supported for now as threshold selection is supervised.
* The `test` set will be used to assess the final performance of the AD pipeline, and should hence be a list of periods containing both normal and anomalous data.

The `data.spark_manager` and `data.otn_manager` modules implement the data-specific loading, labeling and splitting strategies to constitute the sets defined above. 

To run the pipeline step, call the main script from the `data` directory: 

```bash
(anobench) python make_datasets.py [args]
```

Either use the `-h` or `--help` options or see `utils.common` for a complete description of the relevant command-line arguments. 

**`features` - Step #2: transforming the periods to a suited format for modeling.**

The main goal of this step is to constitute the final features that will be used for modeling the normal behavior of the data, but the periods can also be resampled to other sampling periods to decrease the data resolution.

The `features.spark_alteration` and `features.otn_alteration` modules implement the data-specific features alteration functions (*i.e.* features addition, deletion or transformation).

The outputs of this step are the transformed sets periods and labels, saved as `numpy` arrays under `OUTPUTS_ROOT/data/processed`. Any parametric model used for features extraction will be saved under `OUTPUTS_ROOT/models` (*e.g.* PCA model, scaler...). 

To run the step, call the following main script from the `features` directory: 

```bash
(anobench) python build_features.py [args]
```

**`modeling` - Step #3: modeling the normal behavior of the data.**

Modeling the normal behavior of the data is commonly done by training a model to perform a task on normal data only. This model will then be expected to perform this task better under normal inputs (from its training distribution) than under anomalous ones (unseen before). The outlier score of a record or window will hence be typically derived from the task error committed by the model on that record/window. 

In this pipeline step, we focus on the modeling task, independently from the way this task will be used for deriving the outlier scores (which is the focus of the next step).

Like mentioned previously, the modeling step uses the pipeline's `train` periods, which should only contain normal data. In order to validate and test the modeling task, we further split this `train` set into `train/val/test` samples (typically, 70% of the pipeline's `train` set will go to the modeling's `train` set, 15% to its `val` set and the remaining 15% to its `test` set).

The `modeling.data_splitters`, `modeling.spark_splitters` and `modeling.otn_splitters` modules respectively implement the common and data-specific splitting strategies for constituting the `train/val/test` samples.

* `modeling.forecasting.*` - modules for training, tuning and evaluating forecasting models (in case the modeling task is set to be time-series forecasting).
* `modeling.reconstruction.*` - modules for training, tuning and evaluating window reconstruction models (in case the modeling task is set to be window reconstruction).

The outputs of this step are the model, trained on the modeling's `train` set, and its evaluation on the `train`, `val` and `test` sets, both saved under `OUTPUTS_ROOT/models`

To run this step, call the following main script from the `modeling` directory: 

```bash
(anobench) python train_model.py [args]
```

**`scoring` - Step #4: deriving outlier scores for the records based on the model's predictions.**

This is the pipeline step defining the expression of the outlier score for a given record. Examples include its relative forecasting error by a forecasting model, or the mean reconstruction errors of the windows it belongs to by an autoencoder network. 

* `scoring.forecasting.*` - modules for deriving outlier scores from time-series forecasts.
* `scoring.reconstruction.*` - modules for deriving outlier scores from window reconstructions.

If the scoring method is parametric (*e.g.* if it involves fitting an error vectors distribution to further set the scores as negative log-likelihoods with respect to it), it is fit on the modeling's `test` set and saved under `OUTPUTS_ROOT/models`. 

The output of this step is any parametric scoring model along with the evaluation of the scoring performance on the pipeline's `threshold` and `test` periods. This evaluation will reflect the ability of the AD pipeline to separate out normal from anomalous data by the outlier scores, independently from any hard outlier score threshold/classification.  

To run this step, call the following main script from the `scoring` directory: 

```bash
(anobench) python train_scorer.py [args]
```

**`thresholding` - Step #5: calibrating an outlier score threshold to produce hard classifications.**

The classified elements should be point-wise records, as it enables the same AD pipeline to flag anomalous ranges of arbitrary sizes.

For now, only supervised threshold selection is supported in this project, meaning that the selected outlier score threshold will be the one maximizing the AD performance on the `threshold` set. Available performance metrics are defined and explained within the **`metrics`** package.

The outputs of this step are the selected outlier score threshold and the final evaluation of the binary predictions produced by the overall pipeline, again both saved under `OUTPUTS_ROOT/models`.

To run this step, call the following main script from the `thresholding` directory: 

```bash
(anobench) python train_detector.py [args]
```
