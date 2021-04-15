import cv2
import os
import tensorflow as tf
from model.inception import InceptionResNetV2
import numpy as np
from tqdm.auto import tqdm

dirname = './lwf/lwf_cropped'
put_dirname = './lwf/embeddings'

os.mkdir(put_dirname)

face_encoder = InceptionResNetV2()
face_encoder.load_weights('./data/facenet_keras_weights.h5')

for folder in tqdm(os.listdir(dirname)):

    if len(os.listdir(dirname+'/'+folder))==0:
        continue

    os.mkdir(put_dirname+'/'+folder)
    for file in os.listdir(dirname+'/'+folder):
        
        img = cv2.imread(dirname + '/'+folder + '/' + file)
        img_in = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_in = tf.image.resize(img_in,(160,160))
        img_in = tf.expand_dims(img_in,0)
        
        encoding = face_encoder(img_in)[0].numpy()
        np.save(put_dirname+'/'+folder+'/'+file, encoding)
        


