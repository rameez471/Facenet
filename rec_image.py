import tensorflow as tf
import cv2
import os
from model.mtcnn import MTCNN
import numpy as np

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x,x)))

mtcnn = MTCNN()
face_encoder = tf.keras.models.load_model('./data/facenet_keras.h5')

X,y = [],[]
idx = 0
class_name = []
crop_dirname = './data/test/images_cropped'

for folder in os.listdir(crop_dirname):
    class_name.append(folder)
    for file in os.listdir(crop_dirname+'/'+folder):
        img = cv2.imread(crop_dirname + '/'+folder + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.image.resize(img,(160,160))
        img_in *= 1./255
        img_in = tf.expand_dims(img_in,0)
        embedding = face_encoder(img_in).numpy()
        embedding = l2_normalize(embedding)
        X.append(embedding)
        y.append(idx)
    idx += 1

X = np.array(X)
y = np.array(y)

num_images = X.shape[0]
threshold = 0.9

img = cv2.imread('demo_2.jpg')
img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


bboxes, landmarks, scores = mtcnn.detect(img_in)

for box, landmark, score in zip(bboxes, landmarks, scores):
    img_crop = img_in[int(box[1]):int(box[3]),int(box[0]):int(box[2]) , :] 
    img_crop = tf.image.resize(img_crop,(160,160))
    img_crop *= 1./255
    img_crop = tf.expand_dims(img_crop,0)
    unknown = face_encoder(img_crop).numpy()
    unknown = l2_normalize(unknown)
    distances = []
    for i in range(num_images):
        dist = np.linalg.norm(X[i]-unknown)
        distances.append(dist)

    distances = np.array(distances)
    idx = np.argmin(distances)
    print(distances[idx])
    if distances[idx] < threshold:
        
        person = class_name[y[idx]]
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])), (0, 255, 0), 2)
        img = cv2.putText(img,person ,(int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
    else:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])), (0, 0, 255), 2)
        img = cv2.putText(img, 'Unknown',(int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))


cv2.imwrite('Result_2.jpg', img)


cv2.imshow('Result',img)
cv2.destroyAllWindows()