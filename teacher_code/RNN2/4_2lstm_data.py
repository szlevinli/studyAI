import numpy as np 
import matplotlib.pyplot as plt

#training paramters
batch_start= 0
batch_size = 100
step_num = 150

#model parameters
n_input = 1
n_step = 200
n_hidden = 128
n_output = 1

def get_batch():
    global batch_start, n_step
    # xs shape (50batch, 20steps)
    xs = np.arange(batch_start, batch_start+n_step*batch_size).reshape((batch_size, n_step)) / (10*np.pi)
    print(xs.shape)
    seq = np.sin(xs)
    res = np.cos(xs)
    batch_start += n_step
    plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

get_batch()
