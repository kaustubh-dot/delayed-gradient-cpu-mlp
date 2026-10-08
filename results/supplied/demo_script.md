# Demo Notes

## What the project does

This project compares sequential, synchronous, and bounded delayed-gradient training on a small MNIST neural network.

The network maps 784 pixel values through a 32-neuron ReLU hidden layer to 10 digit classes. The experiments use 5,000 training examples per epoch and evaluate on 1,000 test examples.

## Demo commands

Run from the project root in the MSYS2 UCRT64 shell:

```bash
./sequential_baseline.exe 5
./mnist_reader.exe 0 1 5
./mnist_reader.exe 4 8 5