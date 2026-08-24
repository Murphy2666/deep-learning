import pickle
import gzip
import os

import numpy as np

path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "mnist.pkl.gz"
)

# for avg_darkness_classifier.py
def load_data():
    with gzip.open(path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        return training_data, validation_data, test_data

# for run_network.py

# training_data:
# [(),(),...,()]；() means (x,y)；
# x is np.array, shape = (784,1)；y is one-hot, shape = (10,1)
# validation_data, test_data:
# same, but y is a scalar
def wrapper(data, is_training = False):
    inputs = [np.reshape(input,(len(input), 1)) for input in data[0]]
    result = [one_hot(y) if is_training else y for y in data[1]]
    return [(x, y) for (x, y) in zip(inputs, result)]
    # return zip(inputs, result)

def one_hot(y):
    one_hot = np.zeros((10, 1))
    one_hot[y] = 1.0
    return one_hot

def load_wrapper():
    training_data, validation_data, test_data = load_data()
    return wrapper(training_data, is_training = True), wrapper(validation_data), wrapper(test_data)


# see if data shape is right

if __name__ == '__main__':
    training_data, validation_data, test_data = load_data()

    # training_data
    print("-------- for avg_darkness_classifier --------")
    print("-------- training_data --------")
    print(f"training_data[0].shape is: ", training_data[0].shape)
    print(f"training_data[1].shape is: ", training_data[1].shape)

    # validation_data
    print("-------- validation_data --------")
    print(f"validation_data[0].shape is: ", validation_data[0].shape)
    print(f"validation_data[1].shape is: ", validation_data[1].shape)

    # test_data
    print("-------- test_data --------")
    print(f"test_data[0].shape is: ", test_data[0].shape)
    print(f"test_data[1].shape is: ", test_data[1].shape)


    training_data, validation_data, test_data = load_wrapper()
    # load_wrapper()
    print("-------- for network_run --------")

    # training_data
    print("-------- training_data --------")
    print(f"training_data, x shape is: ", training_data[0][0].shape)
    print(f"training_data, y shape is: ", training_data[0][1].shape)
    print(f"training data length is {len(training_data)}")

    # test data
    print("-------- test_data --------")
    print(f"test_data, x shape is: ", test_data[0][0].shape)
    print(f"test_data, y shape is: ", test_data[0][1].shape)







