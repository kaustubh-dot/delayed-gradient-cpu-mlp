# Demonstration

Build both executables and download MNIST using the README commands. Run commands from the project root. On Windows, append `.exe` to the executable paths below.

```bash
./build/sequential_baseline 5
./build/mnist_reader 0 1 5
./build/mnist_reader 0 8 5
./build/mnist_reader 4 8 5
./build/mnist_reader 8 8 5
```

Start with the sequential and synchronous results. Their loss and accuracy checkpoints should agree on the same toolchain. Synchronous execution permits one example in flight, so every returned gradient has age zero.

Next compare zero-age delayed execution with delay four. Both permit eight examples in flight. The zero-age gate discards many hidden gradients, which reduces work but can weaken learning. Delay four accepts more hidden updates and usually gives a better accuracy trajectory in the recorded comparisons. Point out that the output layer still updates for every example.

Finally, show delay eight. Its bound exceeds the largest reachable age for this eight-example FIFO pipeline, so all hidden updates are accepted. Compare its elapsed time against sequential training rather than assuming two workers must be faster.

Use the report figures to compare complete curves, final accuracy, hidden acceptance, and the first 90% checkpoint. Timings and asynchronous acceptance counts may change on another run. The worker input-pop durations describe elapsed queue calls, not CPU usage.
