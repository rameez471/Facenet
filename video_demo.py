import tensorflow as tf
import cv2
import os
from model.inception import InceptionResNetV2
from model.mtcnn import MTCNN
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def distance(embedding_1, embedding_2, distance_metric=0):
    if distance_metric==1:
        diff = np.subtract(embedding_1, embedding_2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==0:
        dot =  np.sum(np.multiply(embedding_1, embedding_2), axis=1)
        norm = np.linalg.norm(embedding_1, axis=1) * np.linalg.norm(embedding_2, axis=1)
        similarity = dot/norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    
    return dist


image_dir = './data/test/images'
crop_dirname = './data/test/images_cropped'

crop_size = 160

mtcnn = MTCNN()
face_encoder = tf.keras.models.load_model('./data/facenet_keras.h5')

X,y = [],[]
idx = 0
class_name = []

for folder in os.listdir(crop_dirname):
    class_name.append(folder)
    for file in os.listdir(crop_dirname+'/'+folder):
        img = cv2.imread(crop_dirname + '/'+folder + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.image.resize(img,(160,160))
        img_in *= 1./255
        img_in = tf.expand_dims(img_in,0)
        embedding = face_encoder(img_in)[0].numpy()
        X.append(embedding)
        y.append(idx)
    idx += 1

X = np.array(X)
y = np.array(y)




param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
svm = GridSearchCV(SVC(C=1,probability=True), param_grid, cv=5).fit(X, y)

x = X[20]
x = x.reshape(1,-1)

print(svm.predict(x))
print(y[20])
print(svm.predict_proba(x))

prob = svm.predict_proba(x)
y_pred = np.argmax(prob)

print(y_pred)

input_path = './data/test/bvs.mp4'
output_path = './data/test/bvs_result.mp4'

vid = cv2.VideoCapture(input_path)
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")

video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
video_fps = vid.get(cv2.CAP_PROP_FPS)
video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print("!!! TYPE: ", type(output_path),type(video_FourCC),type(video_fps),type(video_size))
out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

while True:
    _,img = vid.read()
    if img is None:
        break
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, landmarks, scores = mtcnn.detect(img_in)

    for box, landmark, score in zip(bboxes, landmarks, scores):
        img_crop = img_in[int(box[1]):int(box[3]),int(box[0]):int(box[2]) , :] 
        img_in = tf.image.resize(img,(160,160))
        img_in *= 1./255
        img_in = tf.expand_dims(img_in,0)
        unknown = face_encoder(img_in)[0].numpy()
        prob = svm.predict_proba(unknown.reshape(1,-1))[0]
        index = np.argmax(prob)
        print(prob[index])
        if prob[index] > 0.80:
            person = class_name[index]
            print(prob[index])
            img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (0, 255, 0), 2)
            img = cv2.putText(img,person ,(int(box[0]), int(box[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
        else:
            img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (0, 0, 255), 2)
            img = cv2.putText(img, 'Unknown',(int(box[0]), int(box[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))

    cv2.imshow('demo',img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


# print(np.linalg.norm(embeddings['Bruce_Wayne'] - embeddings['Clark_Kent']))
# print(np.linalg.norm(embeddings['Bruce_Wayne'] - embeddings['Bruce_Wayne_2']))
# print(np.linalg.norm(embeddings['Bruce_Wayne'] - embeddings['Alfred']))
# print(np.linalg.norm(embeddings['Bruce_Wayne_2'] - embeddings['Clark_Kent']))


# print(distance(embeddings['Bruce_Wayne'],embeddings['Clark_Kent']))
# print(distance(embeddings['Bruce_Wayne'],embeddings['Bruce_Wayne_2']))
# print(distance(embeddings['Bruce_Wayne'],embeddings['Alfred']))
# print(distance(embeddings['Bruce_Wayne_2'],embeddings['Clark_Kent']))

