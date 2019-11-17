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
        x {ndarray 1D} -- 集合中属性 a 的第 k 的分类情况。比如：'色泽' 属性，他的 '乌黑' 分类情况是
                          ------------------------
                          色泽    是好瓜     不是好瓜
                          ------------------------
                          乌黑      4          2
                          ------------------------
                          则 x = [4,2]
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
# numpy 库 vectorize 函数的 signature 属性用于说明被封装的函数（这里是 weight_information_entropy)
# 的规范（签名）要求，本例中设置为 (i),()->() 意思是 weight_information_entropy 函数接收两个参数和一个
# 返回结果，其中第一个参数是一维的,，第二个参数是标量，返回值是标量。这样设置的目的是让 weight_information_entropy
# 函数的第一个参数被解析为一维数据，否则默认情况下会按标量处理。
#
# e.g.
#        集合中属性 a 的分类情况。比如：'色泽' 属性，他的分类情况是
#                          ------------------------
#                          色泽    是好瓜     不是好瓜
#                          ------------------------
#                          乌黑      4          2
#                          浅白      1          4
#                          青绿      3          3
#                          ------------------------
#     input  -> ([[4,2],[1,4],[3,3]], 17)
#     output -> [weight_information_entropy([4,2], 17),
#                weight_information_entropy([1,4], 17),
#                weight_information_entropy([3,3], 17)]
#
###############################################################
vec_weight_information_entropy = np.vectorize(
    weight_information_entropy, signature='(i),()->()')


def information_gain(df, index_name):
    """计算信息增益（information gain）

    Arguments:
        df {DataFrame} -- 数据集
        index_name {string} -- 需要计算信息增益的数据集中的字段名称

    Returns:
        float -- 给定字段的信息增益值
    """
    # 对给定的 DataFrame 进行旋转动作，实现分类汇总
    # 索引列为传入的 index_name
    # 交叉列为字段 '好瓜'
    # 数据为 '编号'
    # 聚合函数为 count 计数
    # 输出结果：
    # ------------------------
    #     好瓜
    # 色泽      是        不是
    # ------------------------
    # 乌黑      4          2
    # 浅白      1          4
    # 青绿      3          3
    # ------------------------
    df_ = df.pivot_table(index=[index_name], columns=[
                         '好瓜'], values=['编号'], aggfunc=['count'])
    # 处理空值
    # 分类汇总会出现空值情况，比如 '敲声' 是 '清脆' 没有 '好瓜' 为 '是' 的，将会出现空值
    df_ = df_.fillna(0)
    # 将 DataFrame 转换为 numpy 的 ndarray
    a = df_.values
    # 按行汇总，求得好瓜或坏瓜
    a_sum = a.sum(axis=0)
    D_entropy = vec_information_entropy(a_sum, a_sum.sum()).sum() * -1
    a_entropy = vec_weight_information_entropy(a, a.sum()).sum()
    return D_entropy - a_entropy


def ID3(df, name='', parent_name='', is_root=False, level=0):

    def pt(msg, df):
        print(f'{msg:=^100}')
        print(df)
    
    is_leaf = df.shape[0] <= 2 or df['好瓜'].nunique() == 1
    node_type = ''
    # Node Type
    if is_root:
        node_type = 'Root Node'
    elif is_leaf:
        node_type = 'Leaf Node'
    else:
        node_type = 'Decision Node'

    msg = f'{name} | {node_type} | {parent_name}'

    if is_leaf:
        pt(msg, df)
        return

    column_gains = {}
    for i in df.columns[1:-1]:
        e = information_gain(df, i)
        column_gains[i] = e
    split_attribute = max(column_gains.keys(), key=lambda key: column_gains[key])
    pt(msg, df)
    split_frames = [frame for _, frame in df.groupby(split_attribute)]
    level_ = level + 1
    for idx, frame in enumerate(split_frames):
        new_name = f'{level_:02d}{idx:02d}({split_attribute})'
        ID3(frame, name=new_name, parent_name=name, is_root=False, level=level_)


if __name__ == "__main__":
    df = pd.read_excel('./data/choice_watermelon.xlsx')
    ID3(df, name='Root', parent_name='Null', is_root=True, level=0)
    # print(df['好瓜'].nunique())
    # print(df.shape[0])
    # print(f'df.columns is {df.columns[1:-1]}')
    # column_gains = {}
    # for i in df.columns[1:-1]:
    #     e = information_gain(df, i)
    #     column_gains[i] = e
    #     print(f'{i} information gain is {e}')
    # next_node = max(column_gains.keys(), key=lambda key: column_gains[key])
    # print(f'next node column name is {next_node}')
    # split_frames = [frame for _, frame in df.groupby(next_node)]
    # for frame in split_frames:
    #     print(frame)
