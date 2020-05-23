import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
# 13x13
def yolo_head(feats, anchors, num_classes):
    # 3
    num_anchors = len(anchors)
    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])  / 32

    # 获得x，y的网格
    # (13,13, 1, 2)
    grid_shape = np.shape(feats)[1:3] # height, width
    print(grid_shape)
    grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y],-1)
    print(np.shape(grid))
    # (batch_size,13,13,3,85)
    feats = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    box_xy = (sigmoid(feats[..., :2]) + grid)
    box_wh = np.exp(feats[..., 2:4]) * anchors_tensor
    box_confidence = sigmoid(feats[..., 4:5])
    box_class_probs = sigmoid(feats[..., 5:])

  
    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.ylim(-2,15)
    plt.xlim(-2,15)
    plt.scatter(grid_x,grid_y)
    plt.scatter(5,5,c='black')
    plt.gca().invert_yaxis()


    anchor_left = grid_x - anchors_tensor/2 
    anchor_top = grid_y - anchors_tensor/2 
    print(np.shape(anchors_tensor))
    rect1 = plt.Rectangle([anchor_left[0,5,5,0,0],anchor_top[0,5,5,0,1]],anchors_tensor[0,0,0,0,0],anchors_tensor[0,0,0,0,1],color="r",fill=False)
    rect2 = plt.Rectangle([anchor_left[0,5,5,1,0],anchor_top[0,5,5,1,1]],anchors_tensor[0,0,0,1,0],anchors_tensor[0,0,0,1,1],color="r",fill=False)
    rect3 = plt.Rectangle([anchor_left[0,5,5,2,0],anchor_top[0,5,5,2,1]],anchors_tensor[0,0,0,2,0],anchors_tensor[0,0,0,2,1],color="r",fill=False)

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    ax = fig.add_subplot(122)
    plt.ylim(-2,15)
    plt.xlim(-2,15)
    plt.scatter(grid_x,grid_y)
    plt.scatter(5,5,c='black')
    plt.scatter(box_xy[0,5,5,:,0],box_xy[0,5,5,:,1],c='r')
    plt.gca().invert_yaxis()

    pre_left = box_xy[...,0] - box_wh[...,0]/2 
    pre_top = box_xy[...,1] - box_wh[...,1]/2 

    rect1 = plt.Rectangle([pre_left[0,5,5,0],pre_top[0,5,5,0]],box_wh[0,5,5,0,0],box_wh[0,5,5,0,1],color="r",fill=False)
    rect2 = plt.Rectangle([pre_left[0,5,5,1],pre_top[0,5,5,1]],box_wh[0,5,5,1,0],box_wh[0,5,5,1,1],color="r",fill=False)
    rect3 = plt.Rectangle([pre_left[0,5,5,2],pre_top[0,5,5,2]],box_wh[0,5,5,2,0],box_wh[0,5,5,2,1],color="r",fill=False)

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    plt.show()
    #
feat = np.random.normal(0,0.5,[4,13,13,75])
anchors = [[142, 110],[192, 243],[459, 401]]
yolo_head(feat,anchors,20)