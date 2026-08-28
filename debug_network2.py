import mnist_loader

import network2

import numpy as np

training_data, validation_data, test_data = mnist_loader.load_wrapper()

mini_batch = [(training_data[0][0], training_data[0][1]),
              (training_data[1][0], training_data[1][1])]

xs = np.hstack([x for (x,_) in mini_batch])
ys = np.hstack([y for (_,y) in mini_batch])

network = network2.network([784, 30, 10])

for i in range(100):
    nabla_w, nabla_b = network.update_mini_batch(mini_batch=mini_batch)
    
    network.weights = [w - 0.5/len(mini_batch) * nw for (w, nw) in zip(network.weights, nabla_w)]
    network.biases = [b - 0.5/len(mini_batch) * nb for (b, nb) in zip(network.biases, nabla_b)]

    if i % 20==0:
        print(f"delta_L(output layer) max: {np.max(abs(nabla_b[-1]))}")

        p_ys = network.feed_forward_trained_model(xs)
        cross_entropy = -1/len(mini_batch) * sum(np.sum(ys * np.log(p_ys) + (1-ys)*np.log(1-p_ys), axis = 0))

        l2_loss = sum(sum((p_ys - ys)**2)) / len(mini_batch)
        print(f"Epoch_{i} loss: cross_entropy = {cross_entropy}")
        print(f"l2_loss = {l2_loss}")

    ## EVALUATE change from L2 loss to cross entropy



    