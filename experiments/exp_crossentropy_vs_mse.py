# Run this script from the project root using:
# python -m experiments.exp_crossentropy_vs_mse

from core import mnist_loader
from core import network2_optimized
import matplotlib.pyplot as plt

# load_data
training_data, validation_data, test_data = mnist_loader.load_wrapper()

# train: cross entropy
print("--- Cross Entropy as cost func ---")
net_ce = network2_optimized.network([784, 30, 10], cost=network2_optimized.CrossEntropy)
# evaluation acc
_, _, _, acc_ce = net_ce.SGD(
    training_data=training_data, mini_batch_size=10, epochs=30, eta=0.5, lmbda=5.0,
    evaluation_data=test_data, monitor_evaluation_accuracy=True
)

# train: mse
print("--- Quadratic Cost as cost func ---")
net_quad = network2_optimized.network([784, 30, 10], cost=network2_optimized.QuadaticCost)
# evaluation acc
_, _, _, acc_quad = net_quad.SGD(
    training_data=training_data, mini_batch_size=10, epochs=30, eta=0.5, lmbda=5.0,
    evaluation_data=test_data, monitor_evaluation_accuracy=True
)

# plt
plt.figure(figsize=(10, 5))
plt.plot(acc_ce, label="Cross-Entropy Cost", color="blue", linewidth=2)
plt.plot(acc_quad, label="Quadratic Cost", color="orange", linestyle="--", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Evaluation Accuracy (Correct Count)")
plt.title("Comparison: Cross-Entropy vs. Quadratic Cost")
plt.legend()
plt.grid(True)
plt.show()
