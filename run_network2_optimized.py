import mnist_loader

import network2_optimized

import time

import matplotlib.pyplot as plt

# begin timing
start_time = time.perf_counter()

# load_data
training_data, validation_data, test_data = mnist_loader.load_wrapper()

# train

net = network2_optimized.network([784, 30, 10], cost = network2_optimized.CrossEntropy)
net.large_weight_initializer()

training_cost, training_accuracy, evaluation_cost, evaluation_accuracy = net.SGD(training_data = training_data
        ,mini_batch_size = 30
        ,epochs = 100
        ,eta = 0.5
        ,lmbda = 5.0
        ,evaluation_data = test_data
        ,monitor_training_cost = True
        ,monitor_training_accuracy = True
        ,monitor_evaluation_cost = True
        ,monitor_evaluation_accuracy = True
        )

# end timing
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"total run time: {elapsed_time} scds")


# plt
plt.figure(figsize=(10, 5))
plt.plot(training_accuracy, label="Training Accuracy")
plt.plot(evaluation_accuracy, label="Evaluation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (Correct Count)")
plt.title("Model Accuracy over Epochs")
plt.legend()
plt.show()