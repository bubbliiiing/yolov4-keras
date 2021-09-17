import tensorflow as tf
from keras import backend as K


#---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    #------------------------------------------#
    #   grid_shape指的是特征层的高和宽
    #------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    #--------------------------------------------------------------------#
    #   获得各个特征点的坐标信息。生成的shape为(13, 13, num_anchors, 2)
    #--------------------------------------------------------------------#
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    #---------------------------------------------------------------#
    #   将先验框进行拓展，生成的shape为(13, 13, num_anchors, 2)
    #---------------------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    #---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #---------------------------------------------------#
    feats           = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    #------------------------------------------#
    #   对先验框进行解码，并进行归一化
    #------------------------------------------#
    box_xy          = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh          = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #------------------------------------------#
    #   获得预测框的置信度
    #------------------------------------------#
    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    
    #---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   图片预测
#---------------------------------------------------#
def DecodeBox(outputs,
            anchors,
            num_classes,
            image_shape,
            input_shape,
            #-----------------------------------------------------------#
            #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
            #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
            #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
            #-----------------------------------------------------------#
            anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):

    box_xy = []
    box_wh = []
    box_confidence  = []
    box_class_probs = []
    for i in range(len(outputs)):
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = \
            get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))
    box_xy          = K.concatenate(box_xy, axis = 0)
    box_wh          = K.concatenate(box_wh, axis = 0)
    box_confidence  = K.concatenate(box_confidence, axis = 0)
    box_class_probs = K.concatenate(box_class_probs, axis = 0)

    #------------------------------------------------------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条，因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对其进行修改，去除灰条的部分。 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    #   如果没有使用letterbox_image也需要将归一化后的box_xy, box_wh调整成相对于原图大小的
    #------------------------------------------------------------------------------------------------------------#
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    box_scores  = box_confidence * box_class_probs

    #-----------------------------------------------------------#
    #   判断得分是否大于score_threshold
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        #-----------------------------------------------------------#
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #-----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        #-----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是：框的位置，得分与种类
        #-----------------------------------------------------------#
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out