#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
from nets.yolo4 import yolo_body
from keras.layers import Input
inputs = Input([416,416,3])
model = yolo_body(inputs,3,80)
model.summary()

for i,layer in enumerate(model.layers):
    print(i,layer.name)
