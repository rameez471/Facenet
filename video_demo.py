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
    """
    Function to calcuate distance betweeen two embeddings
    1 -- Eucledian distances between embedding1 and embedding2
    2 -- Cosine similarity between embedding1 and embedding2
    """
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


def l2_normalize(x):
    """
    L2 Normalize the numpy array
    """
    return x / np.sqrt(np.sum(np.multiply(x,x)))

image_dir = './data/test/images'
crop_dirname = './data/test/images_cropped'

crop_size = 160

mtcnn = MTCNN()
face_encoder = tf.keras.models.load_model('./data/facenet_keras.h5')

X,y = [],[]
idx = 0
class_name = []

# Calculating the embeddings for each person and saving for comparing
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
threshold = 0.65

input_path = './data/test/bvs_Trim.mp4'
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

#Looping the image frame by frame until every frame is processed.
while True:
    _,img = vid.read() #Read a frame
    if img is None:
        break
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, landmarks, scores = mtcnn.detect(img_in)

    #Processing every face detected by MTCNN
    for box, landmark, score in zip(bboxes, landmarks, scores):
        img_crop = img_in[int(box[1]):int(box[3]),int(box[0]):int(box[2]) , :] 
        if img_crop.size == 0:
            continue
        img_crop = tf.image.resize(img_crop,(160,160))
        img_crop *= 1./255
        img_crop = tf.expand_dims(img_crop,0)
        unknown = face_encoder(img_crop).numpy()
        unknown = l2_normalize(unknown)
        distances = []

        #Calculating similarity with each person 
        for i in range(num_images):
            dist = np.linalg.norm(X[i]-unknown)
            distances.append(dist)

        distances = np.array(distances)
        idx = np.argmin(distances)
        print(distances[idx])
        # If minimum distance is less than threshold 
        # person from our database detected otherwise
        # face is unknown.
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

    cv2.imshow('demo',img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

