import numpy as np
import random


''' cost func '''
class CrossEntropy:
    @staticmethod
    def fn(p_ys, ys):
        p_ys = np.clip(p_ys, 1e-15, 1-1e-15)
        return np.sum(-ys * np.log(p_ys) - (1-ys) * np.log(1-p_ys))

    def delta(p_y, y, z):
        return (p_y - y)
    

class QuadaticCost:
    @staticmethod
    def fn(p_ys, ys):
        return np.sum(0.5*((p_ys - ys)**2))

    def delta(p_y, y, z):
        return (p_y - y)*sigmoid_prime(z)


class network:
    def __init__(self, sizes, cost = CrossEntropy):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for (x,y) in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for (x,y) in zip(self.sizes[:-1], self.sizes[1:])]

    def feed_forward_trained_model(self, xs):
        for (w, b) in zip(self.weights, self.biases):
            xs = sigmoid(np.dot(w, xs) + b)
        return xs

    def SGD(self, training_data, mini_batch_size, epochs, eta,
            lmbda = 0.0, # L2 regularization
            evaluation_data = None,
            monitor_training_cost = False,
            monitor_training_accuracy = False,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False
            ):
        
        training_cost, training_accuracy = [], []
        evaluation_cost, evaluation_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                nabla_w, nabla_b = self.update_mini_batch(mini_batch)
                self.weights = [(1 - lmbda*eta/len(training_data))* w - eta/len(mini_batch)*n_w for (w, n_w) in zip(self.weights, nabla_w)]
                #nb is the sum of all samples' nb
                self.biases = [b - eta/len(mini_batch) *n_b for (b, n_b) in zip(self.biases, nabla_b)]
            print(f"epoch_{j} training completed") 

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda, convert = False)
                training_cost.append(cost)
                print(f"training_data cost = {cost}")

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert = True)
                training_accuracy.append(accuracy)
                print(f"training_data accuracy = {accuracy} / {len(training_data)}")

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert = True)
                evaluation_cost.append(cost)
                print(f"evaluation_data cost = {cost}")

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert = False)
                evaluation_accuracy.append(accuracy)
                print(f"evaluation_data accuracy = {accuracy} / {len(evaluation_data)}")

        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy


    def update_mini_batch(self, mini_batch):
        activation = np.hstack([x for (x,_) in mini_batch])
        activations = [activation]
        zs = []
        # feedforward
        for (w,b) in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        ys = np.hstack([y if len(y.shape)>0 and y.shape[0]!=1 else one_hot(y) 
                        for (_,y) in mini_batch])

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # backprop
        delta = (self.cost).delta(activations[-1], ys, zs[-1])
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True) / len(mini_batch)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) / len(mini_batch)
        
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = np.sum(delta, axis = 1, keepdims=True) / len(mini_batch)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) / len(mini_batch)

        return nabla_w, nabla_b 


    def accuracy(self, data, convert = False):
        ''' training_data: convert = True, one_hot
            others: convert = False '''
        xs = np.hstack([x for (x,_) in data])
        ys = [y for (_,y) in data]
        p_ys = self.feed_forward_trained_model(xs).transpose()
        p_y_nums = [np.argmax(p_y) for p_y in p_ys]

        if convert:
            y_nums = [np.argmax(y) for y in ys]
        else:
            y_nums = ys
        return sum([p_y_num == y_num for (p_y_num, y_num) in zip(p_y_nums, y_nums)])


    def total_cost(self, data, lmbda, convert = False):
        ''' training_data: convert = False, one_hot
            others: convert = True, a num'''
        xs =  np.hstack([x for (x,_) in data])
        p_ys = self.feed_forward_trained_model(xs)
        if convert:
            ys = np.hstack([one_hot(y) for (_,y) in data])
        else:
            ys = np.hstack([y for (_,y) in data])
        cost = (self.cost).fn(p_ys, ys) / len(data)
        # L2 regularization
        cost += 0.5 * (lmbda / len(data)) * sum(np.sum(w**2) for w in self.weights)
        return cost


''' others '''
def one_hot(y):
    one_hot = np.zeros((10,1))
    one_hot[y] = 1
    return one_hot
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))















    