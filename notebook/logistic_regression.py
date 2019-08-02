#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebook'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Logistic Regression
# 
# 逻辑回归，也称之为`Sigmoid function`，其公式为
# 
# $
# y=\frac{1}{1+e^{-x}}
# $
# 
# 其输出结果为0到1之间数字
# 
# 图形化展示如下：

#%%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)
y = 1 / (1 + np.exp(-x))

plt.axvline(ls='--', color='r')
plt.plot(x, y)


#%% [markdown]
# ## non-convex function
# 
# 对于

#%%
# sample number
m = 100
x = np.random.randint(10, 1000, m)
y = np.random.randint(0, 2, m)
print(f'x={x}')
print(f'y={y}')

theta = np.random.random_sample(10)
print(f'theta={theta}')
J_theta = (1 / (1 + np.exp(-theta * x)))

#%% [markdown]
# logistic regression 所使用的 cost function 与 log 函数有关，这里先初步理解一下 log 函数