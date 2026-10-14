import mnist_loader

import network2_optimized

import time

# begin timing
start_time = time.perf_counter()

# load_data
training_data, validation_data, test_data = mnist_loader.load_wrapper()

# train

net = network2_optimized.network([784, 30, 10], cost = network2_optimized.CrossEntropy)
net.large_weight_initializer()

net.SGD(training_data = training_data
        ,mini_batch_size = 30
        ,epochs = 10
        ,eta = 0.5
        ,lmbda = 5.0
        # ,evaluation_data = test_data
        ,monitor_training_cost = True
        ,monitor_training_accuracy = True
        # ,monitor_evaluation_cost = False
        # ,monitor_evaluation_accuracy = False
        )

# end timing
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"run time: {elapsed_time} scds")

