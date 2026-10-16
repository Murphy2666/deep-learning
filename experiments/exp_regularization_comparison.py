# Run this script from the project root using:
# python -m experiments.exp_regularization_comparison

from core import mnist_loader
from core import network2_optimized
import matplotlib.pyplot as plt

# load_data
training_data, validation_data, test_data = mnist_loader.load_wrapper()

# train: L2 regularization, lmbda = 0.5 
print("--- L2 regularization lmbda = 5.0 ---")
net_l2 = network2_optimized.network([784, 30, 10], cost=network2_optimized.CrossEntropy, regularization = network2_optimized.L2_penalty)
# evaluation acc
_, _, _, acc_l2 = net_l2.SGD(
    training_data=training_data, mini_batch_size=10, epochs=30, eta=0.5, lmbda=5.0,
    evaluation_data=test_data, monitor_evaluation_accuracy=True
)

# train: L1 regularization, lmbda = 0.5 
print("--- L1 regularization lmbda = 5.0 ---")
net_l1 = network2_optimized.network([784, 30, 10], cost=network2_optimized.CrossEntropy, regularization = network2_optimized.L1_penalty)
# evaluation acc
_, _, _, acc_l1 = net_l1.SGD(
    training_data=training_data, mini_batch_size=10, epochs=30, eta=0.5, lmbda=5.0,
    evaluation_data=test_data, monitor_evaluation_accuracy=True
)


# train: lmbda = 0.0
print("--- lmbda = 0.0 ---")
net_base = network2_optimized.network([784, 30, 10], cost=network2_optimized.CrossEntropy)
net_base.large_weight_initializer()
# evaluation acc
_, _, _, acc_base = net_base .SGD(
    training_data=training_data, mini_batch_size=10, epochs=30, eta=0.5, lmbda=0.0,
    evaluation_data=test_data, monitor_evaluation_accuracy=True
)

# plt
plt.figure(figsize=(10, 5))
plt.plot(acc_l2, label="L2 regularization, lmbda = 0.5", color="blue", linewidth=2)
plt.plot(acc_l1, label="L1 regularization, lmbda = 0.5", color="red", linewidth=2)
plt.plot(acc_base, label="Unregularized, lmbda = 0.0", color="orange", linestyle="--", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Evaluation Accuracy (Correct Count)")
plt.title("Comparison: L2 vs. L1 regularization vs. Unregularized Cost")
plt.legend()
plt.grid(True)
plt.show()
