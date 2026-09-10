import numpy as np
import random

class Network:
    def __init__(self, sizes, cost = CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for (x,y) in zip(self.sizes[:-1], self.sizes[1:])] #Xavier
        """ wi~N(0,1), if input size x is too large
        Assuming E(x_i) = 0, Var(x_i) = 1 for inputs (all layers)
        z=sum(wi*xi)+b, i: [1, n^(L-1)] whose var = x*var(wi)*var(xi) = x*1*1 for input layer
        so if x is too big, it will trigger activation (sigmoid) saturation, causing gradient vanishment
        so change w_i to be divided by sqrt(x), then Var(w_i) = 1/x, Var(z) = x*1/x*1 =1
        """
            
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for (x,y) in zip(self.sizes[:-1], self.sizes[1:])]

    def feed_forward_trained_model(self, xs):
        for (w,b) in zip(self.weights, self.biases):
            xs = sigmoid(np.dot(w, xs) + b)
        return xs

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            lmbda = 0.0, 
            evaluation_data = None,
            monitor_training_cost = None,
            monitor_training_accuracy = None,
            monitor_evaluation_cost = None,
            monitor_evaluation_accuracy = None
            ):
        if evaluation_data: n_data = len(evaluation_data)
        training_cost, training_accuracy = [], []
        evaluation_cost, evaluation_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                nabla_w, nabla_b = self.update_mini_batch(mini_batch)
                # l2 regularization, w
                self.weights  = [(1-lmbda*eta/len(training_data))*w - eta*n_w for (w, n_w) in zip(self.weights, nabla_w)]
                self.biases = [b - eta*n_b for (b, n_b) in zip(self.biases, nabla_b)]
            print(f"epoch_{j} completed")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"training_data cost = {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert = True)
                training_accuracy.append(accuracy)
                print(f"training_data accuracy = {accuracy}")
            if monitor_evaluation_cost:
                cost  = self.total_cost(evaluation_data, lmbda, convert = True)
                evaluation_cost.append(cost)
                print(f"evaluation_data cost = {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"evaluation_data accuracy = {accuracy}")

        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy
        
    def update_mini_batch(self, mini_batch):
        """ vectorized update """
        activation = np.hstack([x for (x,_) in mini_batch])
        activations = [activation]
        zs = []
        # feedforward
        for (w,b) in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # y in mini_batch -> 2D matrix
        ys = np.hstack([y if len(y.shape) > 0 and y.shape[0] != 1 else one_hot(y) for (_,y) in mini_batch])
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # back prop
        delta  = (self.cost).delta(zs[-1], activations[-1], ys)
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True) / len(mini_batch)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) / len(mini_batch)
        for l in range(2, len(self.weights) + 1):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / len(mini_batch)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) / len(mini_batch)

        return nabla_w, nabla_b

    def accuracy(self, data, convert = False):
        ''' model gets cost for training_data, accuracy for test, validation 
            training: convert = True, testing: convert = False '''
        xs = np.hstack([x for (x,_) in data])
        ys = [y for (_,y) in data]
        p_ys = self.feed_forward_trained_model(xs).transpose()
        p_y_nums = [np.argmax(p_y) for p_y in p_ys]

        if convert: # training_data
            y_nums = [np.argmax(y) for y in ys]
        else: # validation, test
            y_nums = ys

        return sum([y_num == p_y_num for (y_num, p_y_num) in zip(y_nums, p_y_nums)])

    def total_cost(self, data, lmbda, convert=False):
        ''' training: convert = False, testing: convert = True 
            cuz validation, test not one-hot, reverse to accuracy '''
        xs = np.hstack([x for (x,_) in data])
        p_ys = self.feed_forward_trained_model(xs)
        if convert: 
            ys = np.hstack([one_hot(y) for (_,y) in data])
        else:
            ys = np.hstack([y for (_,y) in data])
            
        cost = (self.cost).fn(p_ys, ys) / len(data)
        # L2 regularization
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost


''' cost func '''
class CrossEntropyCost:

    @staticmethod
    def fn(p_y, y):
        p_y = np.clip(p_y, 1e-15, 1-1e-15) # keep p_y>0 and 1-p_y>0
        return np.sum(-y * np.log(p_y) - (1-y) * np.log(1-p_y))

    @staticmethod
    def delta(z, p_y, y):
        return (p_y - y)

class QuadaticCost:

    @staticmethod
    def fn(p_y, y):
        return np.sum(0.5*(p_y - y)**2)

    @staticmethod
    def delta(z, p_y, y):
        return (p_y - y)*sigmoid_prime(z)

''' others '''
def one_hot(x):
    one_hot = np.zeros((10,1))
    one_hot[x] = 1.0
    return one_hot

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))