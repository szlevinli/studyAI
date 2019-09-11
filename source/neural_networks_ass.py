# pylint: disable=unbalanced-tuple-unpacking
import scipy.io as sio
import numpy as np
from neural_networks import predict


def get_data_from_mat_file(file_path, *args):
    mat = sio.loadmat(file_path)
    return_value = []
    for i in args:
        return_value.append(mat[i])
    return return_value


def get_weights():
    return get_data_from_mat_file('./data/andrew/ex3weights.mat', 'Theta1', 'Theta2')

def get_data():
    return get_data_from_mat_file('./data/andrew/ex3data1.mat', 'X', 'y')

if __name__ == "__main__":
    X, y = get_data()
    theta1, theta2 = get_weights()

    indices = np.random.choice(X.shape[0], 15, replace=False)
    X_ = X[indices, :]
    y_ = y[indices, :]
    y_ = np.reshape(y_, y_.shape[0])

    p = predict(theta1, theta2, X_) + 1

    print(f'{"p is":>18} {p}')
    print(f'{"actural value is":>18} {y_}')
