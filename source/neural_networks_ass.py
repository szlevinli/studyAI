# pylint: disable=unbalanced-tuple-unpacking
import scipy.io as sio
import numpy as np
from neural_networks import predict, nn_cost, nn_gradient
from tools import random_initialize_nn_weights, debug_initialize_nn_weights


def get_data_from_mat_file(file_path, *args):
    mat = sio.loadmat(file_path)
    return_value = []
    for i in args:
        return_value.append(mat[i])
    return return_value


def get_weights(file_path):
    return get_data_from_mat_file(file_path, 'Theta1', 'Theta2')


def get_data(file_path):
    return get_data_from_mat_file(file_path, 'X', 'y')


def predict_digit():
    X, y = get_data('./data/andrew/ex3data1.mat')
    theta1, theta2 = get_weights('./data/andrew/ex3weights.mat')

    indices = np.random.choice(X.shape[0], 15, replace=False)
    X_ = X[indices, :]
    y_ = y[indices, :]
    y_ = np.reshape(y_, y_.shape[0])

    p = predict(theta1, theta2, X_)

    print(f'{"p is":>18} {p}')
    print(f'{"actural value is":>18} {y_}')


def compute_cost_and_gradient():
    X, y = get_data('./data/andrew/ex4data1.mat')
    theta1, theta2 = get_weights('./data/andrew/ex4weights.mat')

    nn_weights = np.concatenate((theta1.flatten(), theta2.flatten()))
    input_layer_size = X.shape[1]
    hidden_layer_size = 25
    num_labels = 10
    lbd = 1

    J = nn_cost(nn_weights, input_layer_size,
                         hidden_layer_size, num_labels, X, y, lbd)

    print(f'J is {J}')

    gradient = nn_gradient(nn_weights, input_layer_size,
                         hidden_layer_size, num_labels, X, y, lbd)
    print(f'Gradient is {gradient}')


if __name__ == "__main__":
    # predict_digit()
    compute_cost_and_gradient()
