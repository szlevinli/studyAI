"""神经网络算法实现
"""

import numpy as np
from logistic_regression import sigmoid


def predict(theta1, theta2, X):
    # add bias unit
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # X shape is (m, n)
    # theta1 shape is (l, n)
    # so (X @ theta1.T) shape is (m, l)
    layer2_z = X @ theta1.T  # shape is (m, l)
    layer2_a = sigmoid(layer2_z)  # shape is (m, l)
    # add bias unit
    layer2_a = np.concatenate(
        (np.ones((layer2_a.shape[0], 1)), layer2_a), axis=1)
    # layer3_z shape is (m, l) * (r, l)
    layer3_z = layer2_a @ theta2.T
    layer3_a = sigmoid(layer3_z)

    return np.argmax(layer3_a, axis=1)
