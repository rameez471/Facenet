import cv2
from model.mtcnn import MTCNN
import os
import tensorflow as tf
from model.inception import InceptionResNetV2
import math
import numpy as np

def distance(embedding_1, embedding_2):
    dot =  np.sum(np.multiply(embedding_1, embedding_2), axis=1)
    norm = np.linalg.norm(embedding_1, axis=1) * np.linalg.norm(embedding_2, axis=1)
    similarity = dot/norm
    dist = np.arccos(similarity) / math.pi
    return dist

face_encoder = InceptionResNetV2()
face_encoder.load_weights('./data/facenet_keras_weights.h5')

mtcnn = MTCNN()

embeddings = {}

for file in os.listdir('The Office'):
    name = file[:int(file.rindex('.'))]
    img = cv2.imread('The Office/'+file)
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.image.resize(img_in,(160,160))
    img_in *= 1./255
    img_in = tf.expand_dims(img_in,0)
    embedding = face_encoder(img_in)[0].numpy()
    embeddings[name] = embedding

input_path = './examples/michael.mp4'
output_path = './examples/rec_result.mp4'

vid = cv2.VideoCapture(input_path)
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")
video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
video_fps = vid.get(cv2.CAP_PROP_FPS)
video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))


print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

threshold = 0.25

while True:
    _, img = vid.read()
    if img is None:
        break

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, landmarks, scores = mtcnn.detect(img_in)
    
    for box in bboxes:
        img_crop = img[int(box[1]):int(box[3]),int(box[0]):int(box[2]) , :]
        img_in = tf.image.resize(img_crop,(160,160))
        img_in *= 1./255
        img_in = tf.expand_dims(img_in,0)
        embedding_1 = face_encoder(img_in)[0].numpy()
        embedding_1 = np.expand_dims(embedding_1,0)
        for person in embeddings.keys():
            embedding_2 = embeddings[person]
            embedding_2 = np.expand_dims(embedding_2,0)
            dist = distance(embedding_1,embedding_2)
            if dist < threshold:
                img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
                img = cv2.putText(img, '{}'.format(person), (int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))


    cv2.imshow('demo',img)
    if output_path is not None:
        out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()