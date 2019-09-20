import numpy as np


def sigmoid(z):
    """返回 sigmoid 函数计算结果

    Arguments:
        z {float | vector | matrix} -- sigmoid 函数的自变量。可以是 scalar, vector, matrix

    Return:
        {float | vector | matrix} -- sigmoid 函数的计算结果。可以是 scalar, vector, matrix
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """返回 sigmoid 函数的导数计算结果

    Arguments:
        z {float | vector | matrix} -- sigmoid 函数的自变量。可以是 scalar, vector, matrix

    Returns:
        {float | vector | matrix} -- sigmoid 函数的计算结果。可以是 scalar, vector, matrix
    """
    s = sigmoid(z)

    return s * (1 - s)


def normalization(arr):
    '''
    正则化
    =====
    采用 （current_value - mean） / std
    std: standard deviation 标准偏差
    按列计算 std mean
    特别注意算法中有除法运算, 需要考虑除零异常问题, 这里采用方法是如果被除数为零, 则将该被除数置为 1,
    返回结果将为 0, 并不会影响结果

    Parameters
    ----------
    s: numpy.ndarray[float]
       需正则化的数据, 支持二维数组

    Returns
    -------
    numpy.ndarray[float]
          正则化后的数据
    '''
    arr_std = np.apply_along_axis(np.std, 0, arr)
    arr_mean = np.apply_along_axis(np.mean, 0, arr)
    # 防止发生除零异常
    arr_std[arr_std == 0] = 1

    return (arr - arr_mean) / arr_std


def map_featrue(X1, X2, degree):
    """特征工程 多项式方法
    将给定的两个特征进行多项式算法, 得到新的特征从而可以更好的拟合数据
    算法为: 假设 degree=i, 则返回的结果为
    X1^i * X2^0, X1^(i-1) * X2^1, X1^(i-2) * X2^2, ..., X^0 * X2^i

    Arguments:
        X1 {ndarray(n, 1) 2D} -- 特征 X1
        X2 {ndarray(n, 1) 2D} -- 特征 X2
        degree {int} -- 多项式的 degree

    Returns:
        ndarray(n, m) -- 生成的新的特征, 其中 m=1+2+...+degree+degree;
        比如: degree = 6, 则 m=1+2+3+4+5+6+6=27
    """
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            temp = (X1 ** (i - j)) * (X2 ** j)
            out = np.concatenate((out, temp), axis=1)
    return out


def random_initialize_nn_weights(input_layer_size, output_layer_size):
    """生成神经网络随机初始化权重
    首先确定一个值 epsilon 其计算公式为 根号6 / 根号(input_layer_size + output_layer_size)
    接着生成平均分布随机数，取值范围在 [0, 1]
    最后使用 [0, 1] * 2 * epsilon - epsilon 来得到随机初始化数据取值范围 [-epsilon, epsilon]

    Arguments:
        input_layer_size {int} -- 权重所关联的映射输入层，注意不是整个神经网络中所指的输入层
        output_layer_size {int} -- 权重所关联的映射输出层，注意不是整个神经网络中所指的输出层

    Returns:
        matrix -- 神经网络初始化参数，shape is (input_layer_size, output_layer_size + 1)
    """
    epsilon = np.sqrt(6) / np.sqrt(input_layer_size + output_layer_size)

    thetas = np.random.rand(output_layer_size, input_layer_size + 1)
    thetas = thetas * 2 * epsilon - epsilon

    return thetas

def debug_initialize_nn_weights(input_layer_size, output_layer_size):
    """生成调试用神经网络初始化权重
    
    Arguments:
        input_layer_size {int} -- 权重所关联的映射输入层，注意不是整个神经网络中所指的输入层
        output_layer_size {int} -- 权重所关联的映射输出层，注意不是整个神经网络中所指的输出层
    
    Returns:
        matrix -- 神经网络初始化参数，shape is (input_layer_size, output_layer_size + 1)
    """
    rows = output_layer_size
    columns = input_layer_size + 1
    x = np.arange(1, rows * columns + 1)
    x = np.reshape(x, (rows, columns))

    return np.sin(x)


def compute_numerical_nn_gradient(J, theta):
    """用于神经网络梯度（backpropagation）检测的数字化梯度计算
    
    Arguments:
        J {Function} -- 神经网络成本函数
        theta {matrix} -- 神经网络权重
    
    Returns:
        matrix -- 数字化梯度
    """
    numerical_gradient = np.zeros(theta.size)
    perturbation = np.zeros(theta.size)
    e = 1e-4

    for i in range(theta.size + 1):
        perturbation[i] = e
        loss1 = J(theta - perturbation)
        loss2 = J(theta + perturbation)
        numerical_gradient[i] = (loss2 - loss1) / (2 * e)
        perturbation[i] = 0
    
    return numerical_gradient
