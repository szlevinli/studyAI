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
# # Cost Function
#
# `cost function` 成本函数，也称之为代价函数、损失函数，在人工智能领域其主要用于评估假设函数 `hypothesis function` $h_\theta(x)$ 的输出值（通常称为**预测值**）与**真实值**之间的差距。通过拟合 `fitting` 数据调整 `hypothesis function` 的系数 $\theta$ 来建立模型。
#
# `cost function` 需要是**凸函数**，这样才能提供梯度下降算法找到最低点，从而得到最小成本的 `hypothesis function` 的系数 $\theta$
#
# `cost function` 成本函数用 $Cost(h_\theta(x))$ 来表示。
#
# 在人工智能领域，运用 `cost function` 的目标是找到 `hypothesis function` 的系数 $\theta$ 使得 `cost function` 取得最小值, 用数学公式可以表示为 
# $$
# J(\theta)=\frac{1}{m}\sum_{i=1}^{m} Cost(h_\theta(x^{(i)}), y^{(i)})
# $$
#
# - $J(\theta)$：在有 `m` 条记录的训练集中，每一条记录的成本（$Cost(h_\theta(x^{(i)}), y^{(i)})$）合计起来然后求平均值，从而得到在给定的 $\Theta^T$下，用 `hypothesis function` 计算出的预测值与实际值的差距。
# - $m$: 训练集的记录数
# - $i$：训练集的记录索引号，比如 $i=3$ 表示第3条记录
# - $Cost(h_\theta(x^{(i)}), y^{(i)})$: 成本函数（不同的算法模型会有不同成本函数，这里所写的是一种数学方式的表示方法）
# - $h_\theta(x^{(i)})$: 假设函数，第 $i$ 条记录的预测值（不同的算法模型会有不同假设函数，这里所写的是一种数学方式的表示方法）
# - $ y^{(i)}$：第 $i$ 条记录的实际值
#
#
# ## Cost Function For Linear Regression
# 
# $$
# J(\theta)=\frac{1}{m}\sum_{i=1}^{m} Cost(h_\theta(x^{(i)}), y^{(i)})
# $$
# $$
# Cost(h_\theta(x^{(i)}), y^{(i)})=\frac{1}{2}((h_\theta(x^{(i)})-y^{(i)})^2
# $$
# $$
# h_\theta(x^{(i)})=\Theta^Tx^{(i)}
# $$
#
#
# ## Cost Function For Logistic Regression
# 
# $$
# J(\theta)=\frac{1}{m}\sum_{i=1}^{m} Cost(h_\theta(x^{(i)}), y^{(i)})
# $$
# $$
# Cost(h_\theta(x^{(i)}), y^{(i)})=
# \begin{cases}
#   -log(h_\theta(x^{(i)})) & \quad \text{if } y^{(i)}=1 \\
#   -log(1-h_\theta(x^{(i)})) & \quad \text{if } y^{(i)}=0
# \end{cases}
# $$
# $$
# h_\theta(x^{(i)})=\frac{1}{1+e^{-\Theta^Tx}}
# $$

#%%
