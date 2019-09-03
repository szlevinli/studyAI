import numpy as np
import pandas as pd
import logistic_regression as lr


def get_data(filepath, X_field_names_arr, y_field_name, **kwargs):
    """从文件中获取数据

    Arguments:
        filepath {string} -- 文件名称
        X_field_names_arr {python array} -- 特征字段列表
        y_field_name {string | int} -- 目标字段名称或索引

    Returns:
        ndarray -- theta 2D
        ndarray -- X 特征数据 2D
        ndarray -- y 目标数据 1D
    """
    # get data from file
    df = pd.read_csv(filepath, **kwargs)
    X = df[X_field_names_arr].values
    y = df[y_field_name].values

    # add default value to X
    temp1 = np.ones((X.shape[0], 1))
    X = np.concatenate((temp1, X), axis=1)

    # initial theta
    theta = np.zeros(X.shape[1])

    return theta, X, y


def main():
    theta, X, y = get_data(
        './data/andrew/ex2data1.csv',
        [0, 1],
        2,
        header=None)
    iterator_num = 1000000
    learning_rate = 0.001
    for _ in range(iterator_num):
        J, gradient = lr.cost_function(theta, X, y)
        theta = theta - learning_rate * gradient
    print(f'J is {J}')
    print(f'gradient is {gradient}')


def use_cost_function():
    theta, X, y = get_data(
        './data/andrew/ex2data1.csv',
        [0, 1],
        2,
        header=None)
    J, gradient = lr.cost_function(theta, X, y)

    print(f'J is {J}')
    print(f'gradient is {gradient}')

def use_scipy_optimize_minimize():
    theta, X, y = get_data(
        './data/andrew/ex2data1.csv',
        [0, 1],
        2,
        header=None)
    theta = np.reshape(theta, (theta.shape[0],))
    from scipy.optimize import minimize
    optimize_result = minimize(lr.cost_function_, theta, args=(X, y), jac=lr.gradient)
    for k, v in optimize_result.items():
        print (f'{k} = {v}')

if __name__ == '__main__':
    # main()
    # use_cost_function()
    use_scipy_optimize_minimize()