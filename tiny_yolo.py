from keras.models import Sequential, Model, load_model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
import numpy as np
import scipy.io
import os
import tensorflow as tf


# Anchor box là các box có kích cỡ xác định từ trước, thay vì dự đoán trực tiếp kích thước bounding box
#        yolo dự đoán sai số các bounding box của vật so với anchor box, từ sai số đó tinhs ra bounding box
# Các sai số mà yolo dự đoán ra với 1 box gồm 5 variable: delta_x, delta_y, delta_w, delta_h, box_confidence
# ANCHORS: kích cỡ 5 anchor box được định nghĩa từ trước
# CLASS: số loại object, trong bài này, mình đặt CLASS = 1 cho bài toán single object detect
# GRID_H, GRID_W = 13, 13 : coi 1 image là 1 grids với 13*13 grid cell
# Box = 5 : số lượng bounding box mà yolo dự đoán trong mỗi grid cell
# output mà yolo dự đoán có kích cỡ (GRID_H *GRID_W*BOX* (4 + 1 + CLASS)) = 13*13*5*(5+ CLASS)

def create_yolo_model():
    GRID_H, GRID_W = 13 , 13
    ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
    ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
    BOX = 5
    CLASS = 1

    model = Sequential()
    model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(416,416,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(0,4):
        model.add(Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))

    model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(Activation('linear'))
    model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS)))

    return model


### y_true: xmin, ymin, xmax, ymax
def custom_loss(y_true, y_pred):
    # CONSTANT 
    ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
    ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
    ANCHORS = np.reshape(ANCHORS, [1,1,1,5,2])
    NORM_H, NORM_W = 416, 416
    GRID_H, GRID_W = 13 , 13
    SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0
    BOX = 5
    CLASS = 1
   
    ### x and y     
    pred_box_xy = tf.sigmoid(y_pred[:,:,:,:,:2])
    
    ### w and h
    pred_box_wh = tf.exp(y_pred[:,:,:,:,2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1,1,1,1,2]))
    
    ###  confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
    
    ### probability (face probability)
        # with multi object detect, use softmax
        # pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    pred_box_prob = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 5]), -1)

    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
    print("Y_pred shape: {}".format(y_pred.shape))
    
    ### Adjust ground truth
    # adjust x and y true
    center_xy = .5*(y_true[:,:,:,:,0:2] + y_true[:,:,:,:,2:4])
    center_xy = center_xy / np.reshape([(float(NORM_W)/GRID_W), (float(NORM_H)/GRID_H)], [1,1,1,1,2])
    true_box_xy = center_xy - tf.floor(center_xy)
    
    # adjust w and h
    true_box_wh = (y_true[:,:,:,:,2:4] - y_true[:,:,:,:,0:2])
    true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(NORM_W), float(NORM_H)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    pred_box_area = pred_tem_wh[:,:,:,:,0] * pred_tem_wh[:,:,:,:,1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh
    
    true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    true_box_area = true_tem_wh[:,:,:,:,0] * true_tem_wh[:,:,:,:,1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh
    
    intersect_ul = tf.maximum(pred_box_ul, true_box_ul) 
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
    
    
    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    print("iou shape: {}".format(iou.shape))
    reduce_max = tf.reduce_max(iou, [3], True)
    print("reduce_max shape: {}".format(reduce_max.shape))

    best_box = tf.equal(iou,reduce_max)
    best_box = tf.to_float(best_box)
    print("best_box shape{}".format(best_box.shape))
    
    true_box_conf = tf.expand_dims(best_box * y_true[:,:,:,:,4], -1)
    true_box_prob = y_true[:,:,:,:,5:]
    
    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
    print("Y_true shape: {}".format(y_true.shape))
    
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor
    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf
    weight_prob = tf.concat(CLASS * [true_box_conf], 4) 
    weight_prob = SCALE_PROB * weight_prob 
    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)
    print("Weight shape: {}".format(weight.shape))
    
    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, GRID_W*GRID_H*BOX*(4 + 1 + CLASS)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)
    return loss
