# Delayed Gradients in CPU Neural Network Training

Operating Systems course project by **Kaustubh Upadhya, Dhruv Pekhale, Jeet Mahajan, and Tejas Tandon**, Indian Institute of Information Technology, Sri City.

This project studies how relaxing synchronization affects a small neural network trained on a CPU. Two C++ workers own separate layers and exchange messages through bounded queues. Several examples can be in flight. The hidden worker accepts a returned gradient only when its version age is within the configured limit.

The submission includes the [IEEE report](report/main.pdf), its LaTeX source, active C++ implementation, raw measurements, publication figures, reproduction scripts, and tests.

## Model and execution

- MNIST, 784 inputs, 32 ReLU hidden units, and 10 output logits.
- Softmax cross-entropy, SGD with learning rate 0.01, and seed 12345.
- A seeded shuffle selects 5,000 training and 1,000 official test examples. The selected training indices are reshuffled each epoch.
- Float parameters, zero biases, and seeded normal initialization.
- Sequential training updates both layers in one thread.
- Synchronous workers use `tau_max=0` and one example in flight.
- Delayed workers use a configurable age gate and admission window. Each queue's capacity equals that window.

The output layer updates for every example. Hidden rejection skips its local backward computation and update. On acceptance, the hidden layer recomputes its ReLU mask at the current parameters. A positive-age update therefore combines an older downstream signal with a current local Jacobian. The report explains this behavior and its limits.

## Build and run

Use C++17, a compiler with thread support, and Python 3.10 or later. GCC commands below work in Linux or an MSYS2 terminal on Windows. Windows executables receive an `.exe` suffix; their compiler runtime must be on `PATH`.

```bash
mkdir -p build
g++ -O2 -std=c++17 -pthread -Wall -Wextra -Wpedantic src/sequential_baseline.cpp src/dense_layer.cpp src/mnist.cpp -o build/sequential_baseline
g++ -O2 -std=c++17 -pthread -Wall -Wextra -Wpedantic src/main.cpp src/worker.cpp src/dense_layer.cpp src/mnist.cpp -o build/mnist_reader
python -m pip install -r requirements.txt
python scripts/download_mnist.py
./build/sequential_baseline 5
./build/mnist_reader 0 1 5
./build/mnist_reader 4 8 5
```

Run from the project root. The downloader verifies MNIST checksums and writes the four IDX files under `data/`. Dataset files and compiled executables are excluded from the repository.

The worker arguments are `tau_max [max_in_flight] [epochs]`. Allowed delay bounds are `0, 1, 2, 4, 8, 16`; allowed windows are `1, 2, 4, 8, 16`; epochs range from 1 to 20. Defaults are an eight-example window and one epoch. The sequential executable accepts an optional epoch count, defaulting to one.

CMake is also supported:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
python -m unittest discover -s tests -v
```

For a Windows multi-configuration CMake generator, build the Release configuration and use its executable paths. The Python tests and experiment script otherwise expect both programs directly under `build/`.

## Results

| Five-epoch comparison | Sequential mean | Delay-four mean | Delay-four hidden acceptance |
|---|---:|---:|---:|
| Recorded runs | 28.279 s | 20.521 s | 70.28% |
| New optimized sweep | 3.494 s | 2.408 s | 75.03% |

The new sweep has three scheduling repetitions per setting. Delay four gives a ratio of mean sequential to delayed time of 1.451 and mean final accuracy of 92.07%, versus 91.4% sequentially. These gains include omitted hidden computation. They do not establish equal-work parallel speedup or a universally optimal delay.

The strict zero-age delayed setting finishes quickly but fails to reach 90% in the new five-epoch runs. Limits eight and sixteen accept all reachable ages for an eight-example FIFO pipeline. Their timing benefit over sequential training is small or absent in the retained comparisons.

Read [results and measurement definitions](docs/results.md) for reach counts, the corrected timing-log labels, and the limits of the evidence. Original machine and build settings for the early measurements were not recorded. Each evidence set is kept separate.

## Reproduce the experiments

```bash
python scripts/run_experiments.py --repeats 3 --epochs 5 --output results/my_run --build-flags="-O2 -std=c++17 -pthread -Wall -Wextra -Wpedantic"
python scripts/check_results.py
python scripts/plot_results.py
python scripts/check_report.py
```

The experiment script retains complete raw logs, combined epoch rows, run summaries, source and dataset hashes, and environment metadata. It refuses to overwrite an existing results directory. Use `--sequential` and `--workers` for alternative executable locations and `--compiler` if GCC is not found as `g++`. Supply the flags actually used to build the binaries with `--build-flags`; the script cannot infer them. The example records the direct GCC build above. A CMake or different compiler build may use different flags.

The checks validate the committed measurements and report. The plotting script regenerates the submitted figures from the retained recorded and reproduction CSVs. It exports vector PDF/SVG and 300-dpi PNG. Bands and whiskers show observed ranges, not confidence intervals. The report uses generated numerical table inputs; compile `report/main.tex` with IEEEtran, BibTeX, and the required LaTeX packages, or with Tectonic.

## Contents

| Path | Contents |
|---|---|
| `src/` | Sequential baseline, layer and MNIST code, messages, bounded queue, and workers |
| `tests/` | Numerical, queue, and runtime checks |
| `results/pilot/` | Repeated one-epoch studies, including delay 16 |
| `results/recorded/` | Main five-epoch CSVs and separate worker timing log |
| `results/reproduction/` | New 24-run sweep and build/environment metadata |
| `results/summary/` | Derived statistics and corrected instrumented rows |
| `figures/` | Report figures in PDF, SVG, and PNG |
| `report/` | Final IEEE PDF, LaTeX source, bibliography, and table inputs |
| `scripts/` | Dataset download, experiment runner, plotting, and results/report checks |
| `docs/` | Results definitions and demonstration sequence |

See the [demonstration guide](docs/demo.md) for the presentation run order.
