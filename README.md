# Deep Learning - MNIST Digit Classification

A from-scratch implementation of a FNN (feedforward neural network) for MNIST digit classification only using **numpy**.

## File Descriptions

* **`mnist_loader.py`**: Data loading and preprocessing module.
* **`network.py`**, **`debug_network.py`**, **`run_network.py`**: Baseline implementation processing training samples individually (one input vector $x$ at a time).
* **`network_optimized.py`**, **`debug_network_optimized.py`**, **`run_network_optimized.py`**: Fully vectorized matrix implementation processing full mini-batches in parallel. Runs ~4x faster than the baseline version.

## File Structure

```text
.
├── data/
│   └── mnist.pkl.gz              # MNIST dataset file
├── avg_darkness_classifier.py    # simple classifier using avg darkness 
├── mnist_loader.py               # Data loading & format 
├── network.py                    # Baseline network (element x)
├── debug_network.py              # Debug Baseline network
├── run_network.py                # Baseline network training
├── network_optimized.py          # Fully vectorized network (matrix mini-batch)
├── debug_network_optimized.py    # Debug optimized network
├── run_network_optimized.py      # Optimized network training
├── network2.py                   # Vectorized network using cross-entropy loss
└── debug_network2.py             # Debug cross-entropy network


