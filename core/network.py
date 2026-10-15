# standard pkg
import random

# others
import numpy as np


class network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for (x,y) in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, mini_batch_size, epochs, eta, test_data = None):
        if test_data: n_test = len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch{j}: {self.evaluate(test_data)} / {n_test} tested correct")
            else:
                print(f"Epoch{j}: completed")

    def update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # deal with training data one by one
        # can be optimized to use matrix, computing them simultaneosly
        for (x, y) in mini_batch:
            delta_nabla_w, delta_nabla_b = self.back_prop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - eta/len(mini_batch)*nw for (w, nw) in zip(self.weights, nabla_w)]
        self.biases = [b - eta/len(mini_batch)*nb for (b, nb) in zip(self.biases, nabla_b)]

    def back_prop(self, x, y):
        activation = x
        activations  = [x]
        zs = []
        # feed forward
        for (w, b) in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # back prop
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_w, nabla_b 
    
    def cost_derivative(self, output_activation, y):
        return (output_activation - y)

    def feed_forward_trained_model(self, a):
        for (w,b) in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        return sum([int(np.argmax(self.feed_forward_trained_model(x)) == y) for (x,y) in test_data])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)* (1.0-sigmoid(z))