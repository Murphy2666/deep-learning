# Run this script from the project root using:
# python -m experiments.exp_regularization_comparison

from core import mnist_loader
from core import network2_optimized
import matplotlib.pyplot as plt

# load_data
training_data, validation_data, test_data = mnist_loader.load_wrapper()

# train: L2 regularization, lmbda = 0.5 
print("--- L2 regularization lmbda = 0.5 ---")
net_de = network2_optimized.network([784, 30, 10], cost=network2_optimized.CrossEntropy)
# evaluation acc
_, _, _, acc_ce = net_de.SGD(
    training_data=training_data, mini_batch_size=10, epochs=30, eta=0.5, lmbda=5.0,
    evaluation_data=test_data, monitor_evaluation_accuracy=True
)

# train: lmbda = 0.0
print("--- lmbda = 0.0 ---")
net_large = network2_optimized.network([784, 30, 10], cost=network2_optimized.CrossEntropy)
net_large.large_weight_initializer()
# evaluation acc
_, _, _, acc_quad = net_large .SGD(
    training_data=training_data, mini_batch_size=10, epochs=30, eta=0.5, lmbda=0.0,
    evaluation_data=test_data, monitor_evaluation_accuracy=True
)

# plt
plt.figure(figsize=(10, 5))
plt.plot(acc_ce, label="L2 regularization, lmbda = 0.5", color="blue", linewidth=2)
plt.plot(acc_quad, label="Unregularized, lmbda = 0.0", color="orange", linestyle="--", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Evaluation Accuracy (Correct Count)")
plt.title("Comparison: L2 regularization vs. Unregularized Cost")
plt.legend()
plt.grid(True)
plt.show()
