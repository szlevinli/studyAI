import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

n_samples = 1000 # 样本数量
n_features = 1  # 特征数量

#* 创建特征数据集 
#? shape(n_samples, n_features) 元素为0到10之间的均分数字
# x = np.linspace(0, 10, n_samples * n_features).reshape(n_samples, n_features)
x = 10 * np.random.random((n_samples, n_features))
# x = np.random.normal(0, 1, (n_samples, n_features))

#* 创建标签
noise = np.random.randn(n_samples, 1)
y = x ** 2 - 0.5 + noise
y = np.sum(y, axis=1)
y = y[:, np.newaxis]

#print(y)

#plt.hist(x[:, 0], edgecolor='black')
plt.plot(x, y, 'o')
plt.show()
