import pickle
import gzip
import os

path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "mnist.pkl.gz"
)

def load_data():
    with gzip.open(path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return training_data, validation_data, test_data


# 执行这个文件时会触发（Python 会把 __name__ 设为 '__main__'），import 这个文件时不会
if __name__=='__main__':
    training_data, validation_data, test_data = load_data()
    # training_data
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