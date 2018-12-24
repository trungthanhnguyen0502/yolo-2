import cv2
import copy
from random import randint
import os
from matplotlib import pyplot as plt
import numpy as np

class WiderPreprocess():
    def __init__(self, anns_dir, img_dir):
        self.anns_dir = anns_dir
        self.img_dir = img_dir
        self.color = (255,255,0)
        self.face_number_max = 4


    # With my problems, I dont use the image have too large face S or too many faces (> 5)
    # you can change this code to get suitable image
    def check_image(self, image_data):
        if(not os.path.exists(image_data['file_name'])):
            return False
        if(image_data['face_number'] > self.face_number_max + 1):
            return False
        if(image_data['face_number'] == 1):
            rect = image_data['face_coordinate'][0]
            img = cv2.imread(image_data['file_name'])
            w = abs(rect[2] - rect[0])
            if( w/img.shape[1] > 0.35):
                return False
        return True


    # anns_dir: annotation file link 
    # img_dir: image directory (WiderDataset/WIDER_TRAIN/images)
    # contained_text: substring that a link have to contain -> to ensure that link is image link
    def get_img_data(self, contained_text = 'jpg'):
        file = open(self.anns_dir, 'r')
        data = []
        lines = file.readlines()
        for i in range(len(lines)):
            if(contained_text in lines[i]):
                new_image = {}
                new_image['file_name'] = self.img_dir + lines[i].replace("\n", "")
                new_image['face_number'] = int(lines[i+1])
                new_image['face_coordinate'] = []

                if(new_image['face_number'] <= self.face_number_max):
                    for j in range(new_image['face_number']):
                        rect = self.get_coordinate(lines[i+2+j].replace("\n", ""))
                        if rect != None:
                            new_image['face_coordinate'].append(rect)
                    if(self.check_image(new_image) == True):
                        data.append(new_image)
        return data 

    def get_coordinate(self, text_data):
        coor = text_data.split(" ")
        result = []
        if(len(coor) > 4):
            result = self.to_x1_y1_x2_y2([int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3])])
            return result
        else:
            return None
        
    # (left, top, w, h) -> (xmin, ymin, xmax, ymax)
    def to_x1_y1_x2_y2(self, rect):
        x2 = rect[0] + rect[2]
        y2 = rect[1] + rect[3]
        return [rect[0], rect[1], x2, y2]

     # rect = [xmin, ymin, xmax, ymax] 
    def img_with_rectangle(self, img, rects):
        img2 = copy.deepcopy(img)
        for rect in rects:
            cv2.rectangle(img2, (rect[0],rect[1]), (rect[2],rect[3]), self.color, 2)
        return img2
  
    def plot_image(self, img):
        plt.figure()
        plt.axis('off')
        img = np.array(img, dtype = np.uint8)
        plt.imshow(img[:, :, :: -1]) 
        plt.show() 

    def plot_img_with_raw_data(self, img_data):
        img = cv2.imread(img_data['file_name'])
        print(img)
        print(img.dtype)

        img = self.img_with_rectangle(img, img_data['face_coordinate'])
        self.plot_image(img)