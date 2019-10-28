# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#softmax
x = np.linspace(-10,10)

#tf.softmax()
y_sigmoid = 1/(1+np.exp(-x))

#tf.nn.tanh()
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

#tf.nn.relu()
y_relu = np.array([0*item  if item<0 else item for item in x ]) 

#tf.nn.leaky_relu
y_leaky_relu = np.array([0.2*item  if item<0 else item for item in x ]) 


fig = plt.figure()
# plot sigmoid
ax = fig.add_subplot(221)
ax.plot(x,y_sigmoid)
ax.grid()
ax.set_title('(a) Sigmoid')

# plot tanh
ax = fig.add_subplot(222)
ax.plot(x,y_tanh)
ax.grid()
ax.set_title('(b) Tanh')

# plot relu
ax = fig.add_subplot(223)
ax.plot(x,y_relu)
ax.grid()
ax.set_title('(c) ReLu')

#plot leaky relu
ax = fig.add_subplot(224)
ax.plot(x,y_leaky_relu)
ax.grid()
ax.set_title('(d) Leaky ReLu')

plt.tight_layout()
plt.show()
