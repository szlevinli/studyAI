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
y = 1 / (1 + np.exp(x * -1))

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
def calc_mean_squared_error(predicts, y):
	result = (predicts - y) ** 2
	result = result / 2
	result = result.mean()

	return result

def calc_cost_linear_regression(theta, x, y):
	# sample numbers
	m = x.size

	predict = theta * x
	return calc_mean_squared_error(predict, y)

def calc_cost_logistic_regression(theta, x, y):
	# sample numbers
	m = x.size

	predict = 1 / (1 + np.exp(-theta * x))
	return calc_mean_squared_error(predict, y)

def calc_cost_linear_regression_test():
	theta = 0.5
	x = np.array([1,2,3,4,5])
	y = np.array([6,7,8,9,10])
	result = round(calc_cost_linear_regression(theta, x, y), 3)
	expect = 21.375

	assert result == expect, f'期望值是 {expect}，实际值是 {result}'

def calc_cost_logistic_regression_test():
	theta = 0.5
	x = np.array([1,2,3,4,5])
	y = np.array([6,7,8,9,10])
	result = round(calc_cost_logistic_regression(theta, x, y), 4)
	expect = 26.8097

	assert result == expect, f'期望值是 {expect}，实际值是 {result}'

calc_cost_linear_regression_test()
calc_cost_logistic_regression_test()

#%%
# sample number
m = 1000
x = np.random.randint(-10, 10, m)
y = np.random.randint(0, 2, m)
# print(f'x={x}')
# print(f'y={y}')

# theta 迭代次数
theta_iterators = 1000
# thetas = np.random.random_sample(theta_iterators) * 100
thetas = np.random.randint(-10, 10, theta_iterators)
J_thetas = np.zeros(theta_iterators)
# print(f'thetas={thetas}, size is {thetas.size}')

for idx, theta in np.ndenumerate(thetas):
	J_thetas[idx] = calc_cost_logistic_regression(theta, x, y)
plt.plot(thetas, J_thetas, '.')

#%%
# sample number
m = 1000
x = np.random.randint(-10, 10, m)
y = np.random.randint(0, 2, m)
# print(f'x={x}')
# print(f'y={y}')

# theta 迭代次数
theta_iterators = 1000
# thetas = np.random.random_sample(theta_iterators) * 100
thetas = np.random.randint(-10, 10, theta_iterators)
J_thetas = np.zeros(theta_iterators)
# print(f'thetas={thetas}, size is {thetas.size}')

for idx, theta in np.ndenumerate(thetas):
	J_thetas[idx] = calc_cost_linear_regression(theta, x, y)
plt.plot(thetas, J_thetas, '.')


#%% [markdown]
# logistic regression 所使用的 cost function 与 log 函数有关，这里先初步理解一下 log 函数

#%% [markdown]
# # about $log(x)$ function

#%%
figure, (ax1, ax2) = plt.subplots(1, 2)

x = np.linspace(0.01, 0.99, 1000)
y = -np.log(x)

ax1.set_title(r'$-log(x)$')
ax1.set_xlabel('x')
ax1.set_ylabel('y', rotation=0)

ax1.axhline(y=0, c='r', ls='--')
ax1.axvline(x=1, c='blue', ls='--')

ax1.plot(x, y, label=r'$log(x)$')

y = -np.log(1-x)

ax2.set_title(r'$-log(1-x)$')
ax2.set_xlabel('x')
ax2.set_ylabel('y', rotation=0)

ax2.axhline(y=0, c='r', ls='--')
ax2.axvline(x=0, c='blue', ls='--')

ax2.plot(x, y, label=r'$-log(1-x)$')

plt.show()


#%%
