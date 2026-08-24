# Deep Learning - MNIST Digit Classification

A from-scratch implementation of a deep neural network for MNIST digit classification only using **numpy**.

## File Descriptions

* **`mnist_loader.py`**: Data loading and preprocessing module.
* **`network.py`**, **`debug_network.py`**, **`run_network.py`**: Baseline implementation processing training samples individually (one input vector $x$ at a time).
* **`network_optimized.py`**, **`debug_network_optimized.py`**, **`run_network_optimized.py`**: Fully vectorized matrix implementation processing full mini-batches in parallel. Runs ~4x faster than the baseline version.
