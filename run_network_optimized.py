import mnist_loader

import network_optimized

import time

# begin timing
start_time = time.perf_counter()

# load_data
training_data, validation_data, test_data = mnist_loader.load_wrapper()

# train
net = network_optimized.network(sizes = [784, 30, 10])
net.SGD(training_data = training_data
        ,epochs = 10
        ,eta = 3.0
        ,mini_batch_size = 32
        ,test_data = test_data)

# end timing
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"run time: {elapsed_time} scds")

