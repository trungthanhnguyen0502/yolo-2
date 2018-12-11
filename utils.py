import numpy as np
import cv2
from bound_box import BoundBox
import pickle
import os
from matplotlib import pyplot as plt

def aug_img(img_data, NORM_H= 416, NORM_W = 416):
    img = cv2.imread(img_data['file_name'])
    if(img is None):
        return None, None
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)

    # translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy : (offy + h), offx : (offx + w)]

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5:
        img = cv2.flip(img, 1)
    img = img / (255)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))

    # fix object's position and size
    bound_box = []

    for rect in img_data['face_coordinate']:
        box = {}
        box['xmin'] = rect[0]
        box['xmin'] = int(box['xmin'] * scale - offx)
        box['xmin'] = int(box['xmin'] * float(NORM_W) / w)
        box['xmin'] = max(min(box['xmin'], NORM_W), 0)

        box['xmax'] = rect[2]
        box['xmax'] = int(box['xmax'] * scale - offx)
        box['xmax'] = int(box['xmax'] * float(NORM_W) / w)
        box['xmax'] = max(min(box['xmax'], NORM_W), 0)

        box['ymin'] = rect[1]
        box['ymin'] = int(box['ymin'] * scale - offy)
        box['ymin'] = int(box['ymin'] * float(NORM_H) / h)
        box['ymin'] = max(min(box['ymin'], NORM_H), 0)

        box['ymax'] = rect[3]
        box['ymax'] = int(box['ymax'] * scale - offy)
        box['ymax'] = int(box['ymax'] * float(NORM_H) / h)
        box['ymax'] = max(min(box['ymax'], NORM_H), 0)
        if flip > 0.5:
            xmin = box['xmin']
            box['xmin'] = NORM_W - box['xmax']
            box['xmax'] = NORM_W - xmin
        bound_box.append(box)
    return img, bound_box



def data_gen(imgs_data, batch_size, NORM_W = 416, NORM_H = 416, GRID_W = 13, GRID_H = 13, BOX=5):
    imgs_len = len(imgs_data)
    shuffled_indices = np.random.permutation(np.arange(imgs_len))
    left = 0
    right = batch_size if batch_size < imgs_len else imgs_len
    CLASS = 1
    x_batch = []
    y_batch = []
  
    for index in shuffled_indices[left:right]:
        img_data = imgs_data[index]
        img, bboxes = aug_img(img_data)
        if(img is not None):
            x_img = img
            y_img = np.zeros((GRID_W, GRID_H, BOX, 5+CLASS))
            for bbox in bboxes:
                center_x = (bbox['xmin'] + bbox['xmax'])/2
                center_x = center_x / (float(NORM_W) / GRID_W)
                center_y = (bbox['ymin'] + bbox['ymax'])/2
                center_y = center_y / (float(NORM_H) / GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
                box = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
               
                if grid_x < GRID_W and grid_y < GRID_H:
                    y_img[grid_y, grid_x, :, 0:4] = BOX * [box]
                    y_img[grid_y, grid_x, :, 4]  = BOX * [1.]
                    y_img[grid_y, grid_x, :, 5]  = 1.0
            x_batch.append(x_img)
            y_batch.append(y_img)

    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def interpret_netout(image, netout, CLASS, GRID_H = 13, GRID_W = 13):
    BOX = 5
    boxes = []
    THRESHOLD = 0.3
    ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
    ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
    
  # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox(CLASS)
                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row,col,b,:5]
                box.col, box.row = col, row
                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                classes = netout[row,col,b,5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > THRESHOLD
                    
    sorted_indices = list(reversed(np.argsort([box.probs for box in boxes])))
    for i in range(len(sorted_indices)):
        index_i = sorted_indices[i]
        if boxes[index_i].probs == 0:
            continue
        else:
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                iou = boxes[index_i].iou(boxes[index_j])
                if iou >= 0.4:
                    boxes[index_j].probs = 0
                elif iou == -1:
                    boxes[index_i].probs = 0
    true_boxs = []
    for box in boxes:
        if box.probs > THRESHOLD:
            try:
                xmin  = int((box.x - box.w/2) * image.shape[1])
                xmax  = int((box.x + box.w/2) * image.shape[1])
                ymin  = int((box.y - box.h/2) * image.shape[0])
                ymax  = int((box.y + box.h/2) * image.shape[0])
                true_boxs.append([xmin, ymin, xmax, ymax, box.probs])
            except Exception as e:
                print("some error")
    return true_boxs

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def plot_image(img):
    plt.figure()
    plt.axis('off')
    img = np.array(img, dtype = np.uint8)
    plt.imshow(img[:, :, :: -1]) 
    plt.show() 


