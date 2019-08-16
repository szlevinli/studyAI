import pandas as pd
import numpy as np


def normalization(s):
    '''
    正则化
    =====
    采用 max - min / mean 算法使数值控制在[-1, 1]之间

    Parameters
    ----------
    s: Pandas.Series
       需正则化的数据集

    Returns
    -------
    Series
          正则化后的数据集
    '''
    v_average = s.mean()
    v_diff = s.max() - s.min()

    # 如果最大值与最小值一样，则返回元素值为0的数据集
    if (v_diff == 0):
        return pd.Series(np.full(len(s.index), 0), index=s.index)

    return s.apply(lambda x: (x-v_average)/v_diff)


def calc_prediction_price(thetas, features):
    '''
    计算预测结果
    =============
    根据给定的系数theta和特征features，通过线性假设函数计算预测结果

    Parameters
    ----------
    thetas :   numpy.ndarray[float]
               线性假设函数的系数数组
    features : numpy.ndarray[float]
               特征字段数组

    Returns
    -------
    float
         预测结果
    '''
    # 如果thetas的维度与features的维度不同，则抛出异常
    if thetas.shape[0] != features.shape[1]:
        raise Exception(
            f'thetas rows is {thetas.shape[0]} not equals features columns {features.shape[1]}')

    return features.dot(thetas).sum()


def calc_J_value(thetas, features, targets):
    '''
    计算成本函数
    ============
    采用 MSE 作为成本函数

    Parameters
    ----------
    thetas:    numpy.ndarray[float]
               线性假设函数的系数数组
    features : numpy.ndarray[float]
               特征字段数组
    targets :  numpy.ndarray[float]
               实际值/目标值

    Returns
    -------
    float
         成本函数计算结果值
    '''
    if thetas.shape[0] != features.shape[1] or features.shape[0] != targets.shape[0]:
        raise Exception(
            f'thetas rows is {thetas.shape[0]}'
            f'not equals features columns {features.shape[1]}'
            f'or features rows {features.shape[0]}'
            f'not equals targets rows {targets.shape[0]}')
    p = features.dot(thetas)
    d = p - targets
    c = d ** 2 / 2
    return c.mean()


def calc_J_partial_derivative(theta_old, predictions, targets, features, learning_rate):
    '''
    计算系数theta
    =============

    Parameters
    ----------
    theta_old : float
                原 theta 值
    predictions : Series
                  预测值
    targets ： Series
               实际值/目标值
    features ： Series
                特征字段的值
    learning_rate : float
                    学习率

    Returns
    -------
    float
         new theta value
    '''
    return (theta_old
            - learning_rate * (predictions * features - targets * features).mean())


# def calc_DG(dataset, thetas, features, target_field_name, learning_rate):
    '''
    梯度下降
    ========

    Parameters
    ----------
    dataset : DataFrame
              数据集
    thetas : list<float>
             系数列表
    features : list<string>
               特征字段列表
    target_field_name : string
                        实际值字段名称
    learning_rate : float
                    学习率

    Returns
    -------
    cf : float
         成本函数计算结果
    thetas_new : list<float>
                 新的系数列表
    '''
    if (len(thetas) != len(features)):
        raise Exception('thetas length != features length',
                        'thetas length != features length')

    predictions = normalization(
        calc_prediction_price(dataset, thetas, features))
    targets = normalization(dataset[target_field_name])

    thetas_new = thetas[:]
    cf = calc_CF(predictions, targets)
    for i, v in enumerate(thetas):
        if (i != 0):
            f = normalization(dataset[features[i]])
        else:
            f = dataset[features[i]]

        thetas_new[i] = calc_theta(
            thetas[i], predictions, targets, f, learning_rate)
    return cf, thetas_new


def normal_equation(dataset, targets):
    X = dataset.values
    y = targets.values
    left = np.linalg.inv(X.T.dot(X))
    right = X.T.dot(y)

    return left.dot(right)


if __name__ == "__main__":
    pass
