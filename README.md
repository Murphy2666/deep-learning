# Deep Learning - MNIST Digit Classification

A from-scratch implementation of a FNN (feedforward neural network) for MNIST digit classification only using **numpy**.

## File Structure
```text
001_classify_digits/
│
├── core/                         
│   ├── __init__.py               
│   ├── mnist_loader.py           # MNIST dataset loader
│   ├── network.py                # Basic FNN 
│   ├── network_optimized.py      # Vectorized version of basic 
│   ├── network2.py               # Extended FNN with cross-entropy, L2 regularization
│   └── network2_optimized.py     # Optimized version of network2 (weight initiate, etc.)
│
├── experiments/                  
│   ├── exp_crossentropy_vs_mse.py        # Cross-Entropy vs. Quadratic Cost 
│   ├── exp_regularization_comparison.py  # L2 Regularization vs. No Regularization 
│   └── exp_weight_init_comparison.py     # Weight initialization impact 
│
├── scripts/                      
│   ├── avg_darkness_classifier.py        # Simple baseline classifier using average pixel darkness
│   ├── run_network.py                    # Runner script for basic network
│   ├── run_network_practise.py           # Practice script
│   └── run_network2_optimized.py         # Runner script for optimized network2
│
├── debug/                        
│   ├── debug_network.py
│   ├── debug_network2.py
│   └── debug_network2_optimized.py
│   ├── debug_network_optimized.py
│
├── data/                         
│   └── mnist.pkl.gz              # Dataset storage
│
├── .gitignore                    
└── README.md