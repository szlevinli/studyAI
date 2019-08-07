#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebook'))
	print(os.getcwd())
except:
	pass

#%% setup
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

print(f'numpy version is {np.version.version}')
print(f'matplotlib version is {mpl.__version__}')

#%% [markdown]
# # Linear Regression
#
# ## 背景
#
# 根据 Andrew NG 在[网易云课堂](https://study.163.com/course/courseMain.htm?courseId=1004570029)的 Meachine Learning 课程，使用 python 来编写 linear regression 的算法，从而更加深刻的理解这些算法模型
#
# ## Linear Hypothesis Function（LHF）
#
# Linear Hypothesis Function 线性假设函数用于 Linear Regression 模型，它假设数据特征（输入）和 target（输出）呈现一种线性关系，可以使用如下数学公式表示：
#
# $$
# h(\theta)=\theta_0+\theta_{1}x_1+...+\theta_{n}x_n
# $$
#
# - $h(\theta)$: 表示预测结果
# - $\theta$: 表示系数，从$\theta_0...\theta_n$，是一个标量
# - $x$: 表示数据特征（输入），从$x_1...x_n$，表示有$n$个特征
#
# ## Cost Function（CF）
#
# Cost Function 成本函数，使用该函数去使得上面的 LHF 达到最佳值，通常使用 Mean Sequared Error (MSE) 平均方差公式作为上面 LHF 的 Cost Function，它的公式如下：
#
# $$
# J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2
# $$
#
# - $J(\theta)$: cost function $J$ 在给定的 $\theta$ 情况下的结果
# - $m$: 样本数量
# - $x^{(i)}$: 训练集的第$i$个输入向量
# - $y^{(i)}$: 训练集的第$i$个输出分类标志（就是target）
# - $\theta$: 选择的参数值或权重（$\theta_0 \theta_1 \theta_2 ...$）
# - $(h_{\theta}(x^{(i)})$: 在给定的 $\theta$ 情况下，训练集的第$i$个样本的预测结果（这里用的就是线性假设函数 $h(x)=\theta_0+\theta_1x$ 计算出的
#
# ## Gradient Descent （GD）
#
# Gradient Descent 梯度下降是一种算法，用于计算 Linear Hypothesis Function 的系数（$\theta$）,使用该算法采用迭代的方式可以计算出 CF 的最小值，因此称为“梯度下降”。
#
# 梯度下降的算法如下：
# 1. 随机假设$\theta$的值，通常设置为0
# 2. 通过 LHF 计算训练集中每个样本的预测值
# 3. 计算 CF 结果（这里用的是 MSE 成本函数），这里用$J(\theta)$表示。$(h_{\theta}(x^{(i)})$表示预测值，$y^{(i)}$表示实际值（target）
# 4. 更新$\theta$的值，更新规则为$\theta=\theta-\alpha\frac{\partial}{\partial \theta}J(\theta)$
# 5. 重复（迭代）执行2-4步，直到 $J(\theta)$ 值不变，或者变化很小为止
#
# 这里重点解释$\frac{\partial}{\partial \theta}J(\theta)$的计算方式，首先介绍表示方法：
# - $x_{j}^{(i)}$：表示第$i$个样本的第$j$个特征的值
# - 预设$x_0=1$，因为在 LHF 中$\theta_0$是没有对应的变量，这里预设一个且设置为1，不会影响 LHF
# - 为了方便计算 Andrew NG 对 MSE 函数做了一点调整，该调整并不会影响整个计算结果，主要是在公式前加了一个$\frac{1}{2}$，使得公式变为$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$
#
# 计算过程：
# - $\frac{\partial}{\partial \theta}J(\theta)=\frac{\partial}{\partial \theta}(\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2)$
# ---
# - $\frac{\partial}{\partial \theta}J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}\frac{\partial}{\partial \theta}(h_{\theta}(x^{(i)})-y^{(i)})^2$
# ---
# - $\frac{\partial}{\partial \theta}J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}2(h_{\theta}(x^{(i)})-y^{(i)})\frac{\partial}{\partial \theta}((h_{\theta}(x^{(i)})-y^{(i)})$
# ---
# - $\frac{\partial}{\partial \theta}J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})\frac{\partial}{\partial \theta}((h_{\theta}(x^{(i)})-y^{(i)})$
# ---
# - $\frac{\partial}{\partial \theta}J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}$
#
# 对于$\theta$的更新规则而言:
# - $\theta_0=\theta_0-\frac{\partial}{\partial \theta_0}J(\theta_0)=\theta_0-\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{0}^{(i)}$
# ---
# - $\theta_1=\theta_1-\frac{\partial}{\partial \theta_1}J(\theta_1)=\theta_1-\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{1}^{(i)}$
# ---
# - ...
# ---
# - $\theta_n=\theta_n-\frac{\partial}{\partial \theta_n}J(\theta_n)=\theta_n-\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{n}^{(i)}$
#
#
# ## Stochastic Gradient Descent （SGD）
#
# 当数据集数量非常大时，梯度下降算法的效率会变得很差，因为它每次迭代都要遍历所有数据。
#
# Stochastic Gradient Descent 随机梯度下降算法用于解决梯度下降算法效率问题，它的算法如下：
# 1. 随机假设$\theta$的值，通常设置为0
# 2. 随机从数据集中选择一条样本，通过 LHF 计算该样本的预测值
# 3. 计算 CF 结果（这里用的是 MSE 成本函数），这里用$J(\theta)$表示。$(h_{\theta}(x^{(i)})$表示预测值，$y^{(i)}$表示实际值（target）
# 4. 更新$\theta$的值，更新规则为$\theta=\theta-\frac{\partial}{\partial \theta}J(\theta)$
# 5. 重复（迭代）执行2-4步，直到 $J(\theta)$ 值不变，或者变化很小为止
#
#
# ## Normal Equation （NE）
#
# 用线性代数的方式计算$\theta$值，该算法无需迭代数据集，一步就可算出结果，但该算法有如下条件制约：
# - 只可用于 linear regression 模型
# - 当数据特征（输入字段数）非常大， 其效率会变得很差。根据 Andrew NG 的建议，超过10000个特征字段就不要使用该算法了
#
# $$
# \theta=(X^TX)^{-1}(X^Ty)
# $$
#
# - $\theta$是向量，其中的元素是 LHF 中的各系数值，这个也是最终计算结果
# - $X$是矩阵，其规模为$m\times n$，其中$m$表示数据集的记录数，$n$表示数据集的特征字段数，其元素为具体的数据集中数据，比如$X_{i,j}$表示第$i$个样本第$j$列特征的数据内容
# - $y$表示target即标签或者说是输出
#%% [markdown]
# ## 数据集
#
# 为了实现并验证 linear regression 模型，需要一个数据集（`data/kc_house_data.csv`），这里采用 kaggle 的 House Sales in King County, USA 房价数据。
#
# 字段说明：
# - id： a notation for a house
# - date： Date house was sold
# - price： Price is prediction target
# - bedrooms： Number of Bedrooms/House
# - bathrooms： Number of bathrooms/House
# - sqft_living： square footage of the home
# - sqft_lot： square footage of the lot
# - floors： Total floors (levels) in house
# - waterfront： House which has a view to a waterfront
# - view： Has been viewed
# - condition： How good the condition is ( Overall )
# - grade： overall grade given to the housing unit, based on King County grading system
# - sqft_above： square footage of house apart from basement
# - sqft_basement： square footage of the basement
# - yr_built： Built Year
# - yr_renovated： Year when house was renovated
# - zipcode： zip
# - lat： Latitude coordinate
# - long： Longitude coordinate
# sqft_living15： Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
# - sqft_lot15： lotSize area in 2015(implies-- some renovations)

#%%
import pandas as pd

df = pd.read_csv('../data/kc_house_data.csv')
df

#%% [markdown]
# 使用 pandas.sample 方法创建训练集，提取整个数据集的 20% 的数据，为了保证训练集的稳定，给定 seed 为 1

#%%
train_set = df.sample(frac=0.2, random_state=1)
train_set

#%% [markdown]
# 使用 pandas.sample 方法创建测试集，提取整个数据集的 30% 的数据，为了保证测试集的稳定，给定 seed 为 2

#%%
test_set = df.sample(frac=0.3, random_state=2)
test_set

#%% [markdown]
# ## 数据集特征分析
#
# 字段 price 作为本数据集的 target 或者称之为 flag。现在需要分析哪些字段可以作为特征，根据经验我们先看看 sqft_living 作为特征字段与 target 呈现那种形式，可以使用图表方式来直观的感受

#%%
import matplotlib.pyplot as plt
import numpy as np

def normalization(s):
    v_average = s.mean()
    v_diff = s.max() - s.min()

    if (v_diff == 0):
        return pd.Series(np.full(len(s.index), 0), index=s.index)

    return s.apply(lambda x: (x-v_average)/v_diff)

def plot(df, field_name, y, style, subplt):
    subplt.plot(normalization(df[field_name]), y, style)
    subplt.set_title(field_name)

def plots(fields, data_set, normal_price, row, column):
    arr = np.array(fields)
    if (arr.size > row*column):
        arr = np.array(fields[:row*column])
    arr.resize(arr.size//row+1, column)
    for i in np.ndindex(arr.shape):
        field = arr[i]
        if (field):
            plot(data_set, arr[i], normal_price, 'g^', axes[i])

ROW_NUM = 3
COL_NUM = 3
normal_price = normalization(train_set.price)
fig, axes = plt.subplots(ROW_NUM, ROW_NUM, figsize=(12, 10))
fig.tight_layout()

fields = ['sqft_living', 'bedrooms', 'bathrooms',           'sqft_lot', 'floors', 'grade',           'sqft_above', 'sqft_basement', 'sqft_living15']
plots(fields, train_set, normal_price, ROW_NUM, COL_NUM)

#%% [markdown]
# 从上面图表可以看出，“sqft_living”，“bathromms”，“sqft_above”，“sqft_living15”这四个字段和价格“price”之间称线下关系较为明显，我们就采用这四个字段作为特征字段来进行建模

#%%
def calc_prediction_price(dataset, thetas, features):
    '''
    计算预测结果
    =============
    根据给定的系数theta和特征features，通过线性假设函数计算预测结果

    Parameters
    ----------
    dataset : DataFrame
              数据集
    thetas : [float]
             线性假设函数的系数数组
    features : [string]
               数据集特征字段数组，与 `thetas` 对应

    Returns
    -------
    Series
          预测结果
    '''
    # 返回结果，初始化为 0 的 Series 对象
    result = pd.Series(np.full(len(dataset.index), 0), index=dataset.index)
    # 将 thetas 数组与 features 数组进行结对操作
    # np.stack((['a', 'b', 'c'], [1, 2, 3]), axis=-1) => [['a', 1], ['b', 2], ['c', 3]]
    v = np.stack((np.array(thetas), np.array(features)), axis=-1)
    # 递归 thetas 数组与 features 数组 计算预测结果
    for theta, feature in v:
        result = result + dataset[feature] * float(theta)
    return result

def calc_CF(predictions, targets):
    '''
    计算成本函数
    ============
    采用 MSE 作为成本函数

    Parameters
    ----------
    predictions : Series
                  预测值
    targets : Series
              实际值

    Returns
    -------
    float
         成本函数计算结果值
    '''
    return ((predictions - targets) ** 2).sum() / len(predictions.index)

def calc_theta(theta_old, predictions, targets, features, learning_rate):
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
    return theta_old - learning_rate * (predictions * features - targets * features).sum() / len(predictions.index)

def calc_DG(dataset, thetas, features, target_field_name, learning_rate):
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
        raise Exception('thetas length != features length', 'thetas length != features length')

    predictions = normalization(calc_prediction_price(dataset, thetas, features))
    targets = normalization(dataset[target_field_name])

    thetas_new = thetas[:]
    cf = calc_CF(predictions, targets)
    for i, v in enumerate(thetas):
        if (i != 0):
            f = normalization(dataset[features[i]])
        else:
            f = dataset[features[i]]

        thetas_new[i] = calc_theta(thetas[i], predictions, targets, f, learning_rate)
    return cf, thetas_new

def normal_equation(dataset, targets):
    X = dataset.values
    y = targets.values
    left = np.linalg.inv(X.T.dot(X))
    right = X.T.dot(y)

    return left.dot(right)


#%%
x = train_set[:]
x['field_x'] = pd.Series(np.full(len(train_set.index), 1), index=train_set.index)
x1 = pd.concat([x['field_x'], x['sqft_living'], x['bathrooms'], x['sqft_above'], x['sqft_living15']], axis=1)

y = x['price']
z = normal_equation(x1, y)
z


#%%
slices = 1000
theta_old = 0.0
target_field_name = "price"
learning_rate = 0.0001

# 为了满足线性假设函数，需要增加一个参数和特征，分布设为 1 和 0
# 增加的这个参数即为线性假设函数的截距，通常命名为theta 0，其值设置为1
virtual_field_name = 'field_x'
train_set[virtual_field_name] = pd.Series(np.full(len(train_set.index), 1), index=train_set.index)
thetas = [0.0, 0.0, 0.0, 0.0, 0.0]
features = [virtual_field_name, "sqft_living", "bathrooms", "sqft_above", "sqft_living15"]

for i in range(15):
    cf, thetas = calc_DG(train_set[:slices], thetas, features, target_field_name, learning_rate)
    print(f'cf = {cf:<25} thetas = {thetas}')


#%%
# Test
def calc_prediction_price_test():
    df = pd.DataFrame({'x': [1, 2, 3],
                     'y': [4, 5, 6],
                     'z': [7, 8, 9]})
    thetas = [0.5, 0.6, 0.7]
    field_names = ['x', 'y', 'z']
    expect = pd.Series([7.8, 9.6, 11.4], dtype='float64')
    result = calc_prediction_price(df, thetas, field_names)

    assert result.equals(expect)

def calc_CF_test():
    predections = pd.Series([0.6, 5.808, 4.3])
    targets = pd.Series([1.5, 2.6, 3.8])
    expect = 3.783754667
    result = round(calc_CF(predections, targets), 9)

    assert expect == result

def calc_theta_test():
    theta_old = 0.0
    predictions = pd.Series([1.8, 2.7, 6.5])
    targets = pd.Series([1, 3, 6])
    features = pd.Series([1.3, 1.5, 1.7])
    learning_rate = 0.1

    expect = -0.048
    result = round(calc_theta(theta_old, predictions, targets, features, learning_rate), 3)

    assert expect == result

# calc_prediction_price_test()
calc_CF_test()
calc_theta_test()

#%% [markdown]
# ## Linear Regression Cost Function
#
# `hypothesis function` is $h_\theta(x)=\Theta^T x$ and $Cost(h_\theta(x),y)=\frac{1}{2}(h_\theta(x)-y)^2$
#
# The Cost Function is
#
# $$
# \begin{align*}
# J(\theta) &= \frac{1}{m}\sum_{i=1}^m Cost(h_\theta(x^{(i)}),y^{(i)}) \\
#           &= \frac{1}{m}\sum_{i=1}^m \frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2 \\
#           &= \frac{1}{m}\sum_{i=1}^m \frac{1}{2}(\Theta^Tx-y^{(i)})^2
# \end{align*}
# $$

#%%
m = 100
x = np.linspace(1, 100, m)
y = np.linspace(100, 150, m)
mm = 100
thetas = np.linspace(-10, 10, mm)
J_thetas = np.zeros(mm)

for idx, val in enumerate(thetas):
    c = (val * x - y) ** 2 / 2
    J_thetas[idx] = c.mean()

plt.plot(thetas, J_thetas, '-')

#%%
