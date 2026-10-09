# Experiment results

The project compares a single-thread baseline with two workers that own separate layers of a 784–32–10 MLP. Every output activation produces an output update. A hidden update is accepted only when its version age is at most `tau_max`; otherwise it is discarded.

Three measurement sets are retained:

| Directory | Measurements | Purpose |
|---|---|---|
| `results/pilot/` | 30 delayed and 15 interleaved one-epoch runs | Early repeated comparisons, including delay 16 |
| `results/recorded/` | Nine five-epoch runs and two separate instrumented runs | Learning curves and the main recorded comparison |
| `results/reproduction/` | 24 five-epoch runs, three per configuration | Optimized build with source, dataset, and environment hashes |

The sets have different experiment dates, horizons, and available build metadata. Their timing rows are summarized separately.

In the recorded five-epoch comparison, sequential training averages 28.279 seconds and delay four averages 20.521 seconds, a ratio of 1.378. Delay four accepts 70.28% of hidden gradients on average. Its first recorded 90% checkpoint averages 8.267 seconds versus 11.128 seconds sequentially.

The new sweep uses an Intel Core i5-12450H, Windows 11 build 26300, and MSYS2 GCC 15.2.0 with `-O2`. Sequential training averages 3.494 seconds and delay four averages 2.408 seconds, a ratio of 1.451. Delay four accepts 75.03% of hidden updates and finishes at 92.07% mean accuracy. Sequential and synchronous learning values match to the six-decimal output precision and both finish at 91.4%.

The speed ratios include the benefit of skipped hidden computation. They do not measure equal-work parallel speedup. Full-acceptance delay eight has little timing advantage in the interleaved one-epoch pilot. A delay bound of four is a useful observed setting, not a universal optimum.

## Reading the logs

`results/recorded/worker_timing_runs.txt` has reversed section headings. Its first block reports `Maximum gradient age: 4` and `Maximum batches in flight: 8`, so it is the delayed run. Its second block reports zero and one, so it is synchronous. The numeric fields identify the configurations. The raw log is retained, and the plotting script extracts those fields into `results/summary/instrumented_comparison.csv`.

Training time sums the epoch training windows. It excludes dataset loading, initialization, shuffling, evaluation, and final shutdown. Worker input times surround the complete queue `pop` call and span the worker lifetime, including possible checkpoint-evaluation gaps. Worker times overlap and cannot be converted into CPU utilization by dividing them by training time.

`time_to_90_percent_seconds` is the cumulative training time at the first qualifying epoch checkpoint. `NA` means no recorded checkpoint reaches 90%. In the new sweep, delay zero reaches in none of three runs and delay one in one of three. Every other setting reaches in all three. Means of target time include reaching runs only; final-time and accuracy means include all runs.

Figure bands and error whiskers show observed minimum and maximum. They are not confidence intervals. The repeated runs use one fixed initialization and subset. Affinity, background load, and power settings were not controlled. Original build and machine metadata for the pilot and recorded runs were not preserved.

## Update semantics

The hidden worker caches the input vector. On acceptance, it recomputes the ReLU mask using its current parameters and combines that local Jacobian with the returned downstream derivative. At positive age, this update need not equal the gradient of the origin forward pass or of the current complete model. The report and source describe this rule explicitly.

With eight examples in flight and the current FIFO protocol, returned gradient age cannot exceed seven. Limits eight and sixteen therefore accept the same reachable ages. Their separate runs can still differ because scheduling changes the interleaving of hidden forwards and updates.
