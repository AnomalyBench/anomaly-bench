Exathlon Raw Dataset
==============================

This folder contains the raw dataset for Exathlon in a compressed format. Once uncompressed, the folder structure should look as follows:

```
.
+-- undisturbed
|	+-- *.csv
|   +-- ...
+-- disturbed
|   +-- bursty_input			
|		+-- *.csv
|   	+-- ...
|   +-- bursty_input_crash			
|		+-- *.csv
|   	+-- ...
|   +-- stalled_input			
|		+-- *.csv
|   	+-- ...
|   +-- cpu_contention		
|		+-- *.csv
|   	+-- ...
|	+-- process_failure   
|		+-- *.csv
|   	+-- ...
+-- ground_truth.csv		
```

## Traces Description

We define a *trace* as the recording of a Spark streaming application's execution during a period of time. Each trace consists of a set of 2,283 metrics recorded at a 1-second resolution and is saved as a CSV file. 

The `undisturbed` folder contains traces of Spark application runs that were not manually disturbed by an event injection.  

The `disturbed` folder contains traces of Spark application runs that were manually disturbed by generating one or more controlled abnormal event(s) during precise periods. Disturbed traces are grouped into subfolders corresponding to their type.

## Ground Truth Table Description

**Note:** The convention used here is slightly different from the one introduced in the data description table.

The ground truth table is provided as the `ground_truth.csv` file. Each row of the table corresponds to an injected anomalous event and specifies the following fields: 

* `trace_id` - the identifier of the trace the event applies to, of the form:

`benchmark_userclicks_{app_id}_{sender_id}_{sender_port}_{input_rate}_{batch_id}_{job_id}_`

Where `app_id` is the identifier of the Spark application; `sender_id` the identifier of the sender node, which sent data from `sender_port` at `input_rate` records/sec throughout the normal execution of the application; and `batch_id`, `job_id` respectively the identifier of the batch of concurrent applications the recorded one belonged to and the identifier it took within that batch. 

- `trace_type` - The trace type, following the naming convention of the `disturbed` subfolders.

* `event_type` - The injected event type, typically the same as the trace type except for `driver_failure`, `executor_failure` and `unknown`.
* `event_details` - Some additional details regarding the injected event. This only applies to process failures (that either affected an executor or the application driver) and CPU contentions (that either had no impact, led to increased processing time or caused the application to crash). The value for this field was left empty when it did not apply. 
* `event_start` - The start Unix timestamp of the injected root cause.
* `event_end` - The end Unix timestamp of the injected root cause.
* `extended_end` - The end Unix timestamp of the extended effect the event had on the application after the end of its root cause injection. This timestamp was set for some anomaly types only, either using domain knowledge or manually by visualizing when particular metrics came back to their normal state. For events that did not yield any extended effect on the trace, this field was also left empty.  
