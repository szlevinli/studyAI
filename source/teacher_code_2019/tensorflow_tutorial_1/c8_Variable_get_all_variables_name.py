'''
.meta文件保存了当前图结构
.index文件保存了当前参数名
.data文件保存了当前参数值
'''

import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join("model", "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
	print("variable_name: ", key)
	# print(reader.get_tensor(key))

