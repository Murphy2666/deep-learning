# Run this script from the project root using:
# python -m scripts.run_network

from core import mnist_loader

from core import network

import time

# begin timing
start_time = time.perf_counter()

# load data
training_data, validation_data, test_data = mnist_loader.load_wrapper()

# train, test
net1 = network.network([784, 30, 10])
net1.SGD(training_data = training_data, 
         mini_batch_size = 30, 
         epochs = 10, 
         eta = 3,
         test_data = test_data)

# end timing
end_time = time.perf_counter()
elasped_time = end_time - start_time
print(f"run time: {elasped_time} scds")
