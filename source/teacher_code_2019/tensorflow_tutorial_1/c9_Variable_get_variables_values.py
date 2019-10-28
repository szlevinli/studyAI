'''
.meta文件保存了当前图结构
.index文件保存了当前参数名
.data文件保存了当前参数值
'''

from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join( "model/model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#var_to_shape_map = reader.get_variable_to_shape_map()
tensor = reader.get_tensor("v1")
#for key in var_to_shape_map:
#  print("vairable_name",key)

print(tensor)      
#[[-0.9613502  -0.7398253   1.1505105 ]
# [-0.5695257  -0.03345846 -1.2903801 ]]


tensor = reader.get_tensor("v2")
#for key in var_to_shape_map:
#  print("vairable_name",key)
print(tensor)    

#[[ 1.6577548  -0.14928423]
# [ 0.7018916   0.6668785 ]
# [ 1.3739105   1.5169586 ]]
