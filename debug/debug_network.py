# Run this script from the project root using:
# python -m debug.debug_network

import numpy as np

from core import mnist_loader
from core import network

training_data, validation_data, test_data = mnist_loader.load_wrapper()


x_train, y_train = training_data[0][0], training_data[0][1]
net = network.network([784, 30, 10])

# use just one sample, epochs = 100, see if loss->0
# eta = 3.0
for i in range(100):
    delta_w, delta_b = net.back_prop(x_train, y_train)
    net.weights = [w - d_w * 3.0 for (w, d_w) in zip(net.weights, delta_w)]
    net.biases = [b - d_b * 3.0 for (b, d_b) in zip(net.biases, delta_b)]

    if i%20 ==0:
        loss = sum((net.feed_forward_trained_model(x_train) - y_train) **2)
        print(f"epoch{i}: l2 loss = {loss}")
    