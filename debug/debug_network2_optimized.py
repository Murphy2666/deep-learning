# Run this script from the project root using:
# python -m debug.debug_network2_optimized

import numpy as np

from core import mnist_loader
from core import network2_optimized

training_data, validation_data, test_data = mnist_loader.load_wrapper()
# 2 samples: [(x, y), (x, y)], both train and test
# loss should be decreasing to 0

mini_batch = [(training_data[0][0], training_data[0][1])]

mini_batch = [(training_data[0][0], training_data[0][1])
             ,(training_data[1][0], training_data[1][1])]


xs = np.hstack([x for (x,_) in mini_batch])
ys = np.hstack([y for (_,y) in mini_batch])

net = network2_optimized.network([784, 30, 10], cost = network2_optimized.CrossEntropy)
net.large_weight_initializer()

for i in range(1000):
    nabla_w, nabla_b = net.update_mini_batch(mini_batch = mini_batch)
    net.weights = [w - 3.0/len(mini_batch) * nw for (w, nw) in zip(net.weights, nabla_w)]
    net.biases = [b - 3.0/len(mini_batch) * nb for (b, nb) in zip(net.biases, nabla_b)]

    if i%20 == 0:
        p_ys = net.feed_forward_trained_model(xs)
        loss = network2_optimized.CrossEntropy.fn(p_ys, ys)
        print(f"Epoch{i}: l2_loss = {loss}")



