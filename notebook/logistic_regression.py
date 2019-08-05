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

#%% plot sigmoid function/logistic function
x = np.linspace(-5, 5, 1000)
y = 1 / (1 + np.exp(-x))

plt.axvline(ls='--', color='r')
plt.plot(x, y)


#%% [markdown]
# ## non-convex function
# 
# 对于
# $$
# J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}(\frac{1}{1+e^{-\theta x}} - y)^2
# $$

#%%
def calc_JTheta_linear_regression(theta, x, y):
	# sample numbers
	m = x.size

	result = theta * x
	result = (result - y) ** 2
	result = result / 2
	result = result.mean()

	return result

def calc_JTheta(theta, x, y):
	# sample numbers
	m = x.size

	result = 1 / (1 + np.exp(-theta * x))
	result = (result - y) ** 2
	result = result / 2
	result = result.mean()

	return result

def calc_JTheta_test():
	theta = 0.5
	x = np.array([1,2,3,4,5])
	y = np.array([6,7,8,9,10])
	result = calc_JTheta(theta, x, y)
	expect = 26.8097

	assert round(result, 4) == expect, f'期望值是 {expect}，实际值是 {round(result, 4)}'

# calc_JTheta_test()

#%%
# sample number
m = 1000
x = np.random.randint(-10, 10, m)
y = np.random.randint(0, 2, m)
print(f'x={x}')
print(f'y={y}')

# theta 迭代次数
theta_iterators = 1000
# thetas = np.random.random_sample(theta_iterators) * 100
thetas = np.random.randint(-100, 100, theta_iterators)
J_thetas = np.zeros(theta_iterators)
print(f'thetas={thetas}, size is {thetas.size}')

for idx, theta in np.ndenumerate(thetas):
	J_thetas[idx] = calc_JTheta(theta, x, y)
	print(f'theta={theta}, J_thetas[{idx}]={J_thetas[idx]}')
print(f'J_thetas={J_thetas}')

plt.plot(thetas, J_thetas, '.')

#%% [markdown]
# logistic regression 所使用的 cost function 与 log 函数有关，这里先初步理解一下 log 函数

#%%
