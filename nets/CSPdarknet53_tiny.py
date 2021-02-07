from functools import wraps

import tensorflow as tf
from keras import backend as K
from keras.layers import (Add, Concatenate, Conv2D, Lambda, Layer,
                          MaxPooling2D, UpSampling2D, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

#--------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#   测试中发现没有l2正则化效果更好，所以去掉了l2正则化
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

'''
                    input
                      |
            DarknetConv2D_BN_Leaky
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
            DarknetConv2D_BN_Leaky          |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1    DarknetConv2D_BN_Leaky          |
    |                 |                     |
    -------------Concatenate                |
                      |                     |
        ----DarknetConv2D_BN_Leaky          |
        |             |                     |
      feat       Concatenate-----------------
                      |
                 MaxPooling2D
'''
#---------------------------------------------------#
#   CSPdarknet_tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
def resblock_body(x, num_filters):
    # 利用一个3x3卷积进行特征整合
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    # 引出一个大的残差边route
    route = x

    # 对特征层的通道进行分割，取第二部分作为主干部分。
    x = Lambda(route_group,arguments={'groups':2, 'group_id':1})(x) 
    # 对主干部分进行3x3卷积
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    # 引出一个小的残差边route_1
    route_1 = x
    # 对第主干部分进行3x3卷积
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    # 主干部分与残差部分进行相接
    x = Concatenate()([x, route_1])

    # 对相接后的结果进行1x1卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    feat = x
    x = Concatenate()([route, x])

    # 利用最大池化进行高和宽的压缩
    x = MaxPooling2D(pool_size=[2,2],)(x)

    return x, feat

#---------------------------------------------------#
#   CSPdarknet_tiny的主体部分
#---------------------------------------------------#
def darknet_body(x):
    # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
    # 416,416,3 -> 208,208,32 -> 104,104,64
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(32, (3,3), strides=(2,2))(x)
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(64, (3,3), strides=(2,2))(x)
    
    # 104,104,64 -> 52,52,128
    x, _ = resblock_body(x,num_filters = 64)
    # 52,52,128 -> 26,26,256
    x, _ = resblock_body(x,num_filters = 128)
    # 26,26,256 -> x为13,13,512
    #           -> feat1为26,26,256
    x, feat1 = resblock_body(x,num_filters = 256)
    # 13,13,512 -> 13,13,512
    x = DarknetConv2D_BN_Leaky(512, (3,3))(x)

    feat2 = x
    return feat1, feat2

