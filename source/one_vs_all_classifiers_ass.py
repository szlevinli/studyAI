import numpy as np
import scipy.io as sio
from one_vs_all_classifiers import one_vs_all

if __name__ == "__main__":
    file_path = './data/andrew/ex3data1.mat'
    mat = sio.loadmat(file_path)
    X = mat['X']
    y = mat['y']
    y = np.reshape(y, y.shape[0])
    lbd = 0.1

    all_theta = one_vs_all(X, y, 10, lbd)
    print(f'all_theta is {all_theta}')

    x1 = np.insert(X[0, :], 0, 1)
    result = all_theta @ x1
    print(f'x1 predict is {result}')
