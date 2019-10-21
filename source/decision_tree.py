import numpy as np
import pandas as pd


def information_entropy(x, y):
    """集合中第 k 类的信息熵
    formula = p_k * long_2(p_k)
    p_k = x / y
    p_k 是 当前样本集第 k 类样本所占比率

    Parameters
    ----------
    x {float} -- 第 k 类样本数量
    y {float} -- 当前样本集数量

    Returns
    -------
    {float} -- 第 k 类样本的信息熵
    """
    # 当 k 分类样本数为零时，直接返回 0
    return 0 if x/y == 0 else (x/y) * np.log2(x/y)


###############################################################
#
# 向量化 information_entropy 函数，求集合中第 k 类 信息熵
#
# vec_information_entropy 函数可用于数组，在数组的每个元素上执行
# information_entropy 函数
#
# e.g.
#     input -> ([2, 3], 5)
#     output -> [information_entropy(2, 5), information_entropy(3, 5)]
#
###############################################################
vec_information_entropy = np.vectorize(information_entropy)


def weight_information_entropy(x, x_sum):
    """集合中 a 属性的权重信息熵
    
    Arguments:
        x {ndarray 2D} -- 集合中属性 a 的分类情况。比如：'色泽' 属性，他的分类情况是
                          ------------------------
                          色泽    是好瓜     不是好瓜
                          ------------------------
                          乌黑      4          2
                          浅白      1          4
                          青绿      3          3
                          ------------------------
                          x = [[4,2],[1,4],[3,3]]
        x_sum {float} -- 集合中属性 a 的分类数合计，比如上面的列子，x_sum = 17
    
    Returns:
        float -- 集合中 a 属性的权重信息熵
    """
    # 属性 a 的信息熵
    entropy = vec_information_entropy(x, x.sum()).sum() * -1
    # 属性 a 的权重
    probability = x.sum() / x_sum
    # 集合中 a 属性占整个集合的权重信息熵
    return probability * entropy

###############################################################
#
# 向量化 weight_information_entropy 函数，求集合中 a 属性的权重信息熵
#
# vec_weight_information_entropy 函数可用于数组，在数组的每个元素上执行
# weight_information_entropy 函数
#
# e.g.
#     input -> ([2, 3], 5)
#     output -> [information_entropy_2(2, 5), information_entropy_2(3, 5)]
#
###############################################################
vec_weight_information_entropy = np.vectorize(weight_information_entropy, signature='(i),()->()')

def information_gain(df, index_name):
    df_ = df.pivot_table(index=[index_name], columns=[
                         '好瓜'], values=['编号'], aggfunc=['count'])
    df_ = df_.fillna(0)
    a = df_.values
    a_sum = a.sum(axis=0)
    D_entropy = vec_information_entropy(a_sum, a_sum.sum()).sum() * -1
    a_entropy = vec_weight_information_entropy(a, a.sum()).sum()
    return D_entropy - a_entropy


if __name__ == "__main__":
    df = pd.read_excel('./data/choice_watermelon.xlsx')
    print(f'df.columns is {df.columns[1:-1]}')
    for i in df.columns[1:-1]:
        e = information_gain(df, i)
        print(f'{i} information gain is {e}')

