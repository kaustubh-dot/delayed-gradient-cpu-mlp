# Reproducibility and provenance

## Two independent studies

No original source folder was available. `src/main.cpp` is a newly created reference reconstruction, not the executable that produced the supplied CSVs. The report explicitly separates the old data from the new study. The same architecture and nominal hyperparameters do not establish identical initialization, subset, shuffle sequence, loop order, floating-point behavior, or scheduling.

Original evidence: nine five-epoch CSV runs, 45 checkpoints, plus two separate instrumented five-epoch runs in the timing log. `curves.xlsx` matches the 45 epoch CSV rows to 1e-6. The original guides mention delay 16 in an earlier sweep but provide no numeric results for it.

New evidence: 24 runs, 120 checkpoints, three repetitions of each of eight settings. Each repetition uses a fresh process with seed 12345 and configurations in randomized serial order. These are scheduling repetitions, not a multi-seed generalization study. The host was an Intel Core i5-12450H with 12 logical processors, Windows 11 build 26300, GCC 15.2.0 (MSYS2 MinGW), compiled at `-O2`. The complete metadata and source/binary/data SHA-256 hashes are in `results/fresh/metadata.json`.

## Exact reference semantics

Select the first 5,000 train and first 1,000 official test records. Normalize each byte by 255.0. Shuffle selected train indices each epoch using persistent `std::mt19937(12345)`. Hidden initialization uses an independent generator at 12345 with normal standard deviation sqrt(2/784); output uses seed 12346 and sqrt(1/32). Standard-library normal distribution implementations can differ across toolchains.

An activation stores the current hidden accepted-update counter. Returned hidden age is current counter minus origin. Accept iff age <= tau; increment the counter only for accepted updates. Output always updates. The real output derivative uses pre-update output weights. Hidden backpropagation uses the saved forward activation mask and original immutable input. No mini-batch accumulation is used.

Queues have capacity 2I. The coordinator's unfinished count never exceeds I. Only each layer's owner writes its parameter arrays. An epoch drains all results before evaluation, establishing quiescence and queue-mutex ordering for model reads. Normal shutdown closes the hidden inbox, drains/joins Worker 0, and closes/drains/joins the output stage.

## Timing scopes

Training time is a sum of `steady_clock` windows from training admission through the last result in each epoch. It includes coordination and message transfer; excludes thread creation, dataset load, initialization, evaluation, and final shutdown. This is training-window time, not end-to-end user latency.

Worker active time spans message processing and queue pushes. Input-pop duration spans the full worker lifetime and includes periods during coordinator evaluation and initial/drain waits. Times overlap across threads. Pop duration includes mutex acquisition and call overhead, and is not a direct count of OS sleep states. Push-call duration similarly includes mutex/container overhead; nonzero duration does not prove full-queue blocking. Queue maxima are observed occupancy high-water marks.

Time to target is the first recorded checkpoint with accuracy >=90%. A blank/NR means no checkpoint reached it during five epochs. No interpolation is performed and target crossings may not persist at later checkpoints.

## What cannot be recovered

Original source, compiler flags/version, CPU model, original queue capacity, exact selected record identities, background load, and detailed timer boundaries cannot be recovered by running the new program. The new metadata applies only to the reconstruction. Peak memory, CPU utilization, energy, cache misses, context switches, and time-weighted queue distributions were not measured in either submitted study.

Roll numbers, instructor name, and course code were not supplied and are omitted from the paper rather than invented. The four author names and institution come directly from the user.

## Extending results

Run into a new directory, preserve all stdout/stderr, record source/build/data hashes, and retain the same metric definitions. For a serious follow-up, add more seeds, longer timing windows, validation-based delay selection, pinned affinity/power controls, matched-update sequential controls, and balanced stages. Changes to queue policy, ordering, or number of output workers require a new age/liveness argument.
