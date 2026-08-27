# change loss from L2 to cross entropy

import random

import numpy as np

class network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """ must apply `randn` (mean 0, std 1) instead of `rand` (all positive [0, 1)

         Using `rand` causes z = Wx + b to become too large for high-dimensional inputs. 
         during back prop sigmoid(z) -> 1, sigmoid_prime(z) -> 0
         making delta too small: vanishing gradient """
        self.weights = [np.random.randn(y,x) for (x,y) in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]

    def SGD(self, training_data, epochs, eta, mini_batch_size, test_data = None):
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                nabla_w, nabla_b = self.update_mini_batch(mini_batch)
                self.weights = [w - eta/len(mini_batch) * nw for (w, nw) in zip(self.weights, nabla_w)]
                self.biases = [b - eta/len(mini_batch) * nb for (b, nb) in zip(self.biases, nabla_b)]

            if test_data:
                print(f"Epoch_{j}: {self.evaluate(test_data)} / {len(test_data)} tested correct")
            else:
                print(f"Epoch_{j}: completed")

    def update_mini_batch(self, mini_batch):
        # forward passing
        activation = np.hstack([x for (x,_) in mini_batch])
        activations = [activation]
        zs = []
        for (w, b) in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # back prop
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        ys = np.hstack([y for (_,y) in mini_batch])
        delta = activations[-1] - ys # use cross entropy as loss
        nabla_b[-1] = np.sum(delta, axis = 1, keepdims = True) 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = np.sum(delta, axis = 1, keepdims = True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_w, nabla_b

    def evaluate(self, test_data):
        xs = np.hstack([x for (x, _) in test_data])
        ys = np.hstack([y for (_, y) in test_data])
        p_ys = [np.argmax(y) for y in self.feed_forward_trained_model(xs).transpose()]
        return sum([p_y == y for(p_y, y) in zip(p_ys, ys)])

    def feed_forward_trained_model(self, xs):
        for (w,b) in zip(self.weights, self.biases):
            xs = sigmoid(np.dot(w, xs) + b)
        return xs

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
