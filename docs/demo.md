# Short demo

Build at `-O2`, run `--self-test`, and download MNIST first. Run from the repository root. Use `mlp.exe` on Windows.

```bash
./build/mlp --mode sequential --epochs 5
./build/mlp --mode workers --tau 0 --in-flight 1 --epochs 5
./build/mlp --mode workers --tau 4 --in-flight 8 --epochs 5
```

Explain the architecture while the runs execute: Worker 0 owns the hidden layer, Worker 1 owns the output layer, queues carry activations and true derivatives, and origin versions let Worker 0 reject old hidden updates. Evaluation happens only after an epoch drain.

Show `figures/time_to_target.png` and both original/fresh tradeoff plots. The original data favors delay four; the fresh optimized study does not beat sequential on average. Delayed execution beats the synchronous worker handshake in the fresh study, while tight bounds fail the quality target despite fast completed-example rates.

Do not equate tau=0/I=8 with synchronous tau=0/I=1. Do not call a queue-pop timer CPU utilization or claim delay four is universally optimal. Explain that rejection skips hidden computation and changes the learning stream.
