import tensorflow as tf
import cv2
import sys
import os
from model.mtcnn import MTCNN
import matplotlib.pyplot as plt

dirname = './lwf/lfw-funneled/lfw_funneled'
put_dirname = './lwf/lwf_cropped'
os.mkdir(put_dirname)

print(os.listdir('.'))

crop_width = 160

mtcnn = MTCNN()

files_types = ['pairs.txt','pairs_01.txt','pairs_02.txt','pairs_03.txt','pairs_04.txt','pairs_05.txt','pairs_06.txt','pairs_07.txt','pairs_08.txt','pairs_09.txt','pairs_10.txt']

total_num = 0
cropped_num = 0

for folder in os.listdir(dirname):
    os.mkdir(put_dirname+'/'+folder)
    for file in os.listdir(dirname+'/'+folder):
        img = cv2.imread(dirname+'/'+folder+'/'+file)
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        total_num += 1
        bboxes, landmarks, scores = mtcnn.detect(img_in)
        try:
            box = bboxes[0]
        except:
            continue
        img_crop = img_in[int(box[1]):int(box[3]),int(box[0]):int(box[2]) , :] 
        try:
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
        except:
            continue
        cv2.imwrite(put_dirname+'/'+folder+'/'+file, img_crop)
        cropped_num += 1


print('########################################')
print('{} Images cropped out of {} Images'.format(cropped_num, total_num))
print('########################################')
