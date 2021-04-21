import tensorflow as tf
import cv2
import os
from model.inception import InceptionResNetV2
from model.mtcnn import MTCNN
import numpy as np
import math

def distance(embedding_1, embedding_2, distance_metric=0):
    if distance_metric==0:
        diff = np.subtract(embedding_1, embedding_2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
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
face_encoder = InceptionResNetV2()
face_encoder.load_weights('./data/facenet_keras_weights.h5')


# for file in os.listdir(image_dir):
#     img = cv2.imread(image_dir+'/'+file)
#     img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     bboxes, landmarks, scores = mtcnn.detect(img_in)
#     box = bboxes[0]
#     img_crop = img_in[int(box[1]):int(box[3]),int(box[0]):int(box[2]) , :] 
#     img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
#     img_crop = cv2.resize(img_crop,(crop_size,crop_size))
#     cv2.imwrite(crop_dirname+'/'+file, img_crop)

# print('Images Cropped!!!') 

embeddings = {}

for file in os.listdir(crop_dirname):
    img = cv2.imread(crop_dirname+'/'+file)
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.image.resize(img_in,(160,160))
    img_in *= 1./255
    img_in = tf.expand_dims(img_in,0)
    name = file[:file.rindex('.')]
    encoding = face_encoder(img_in).numpy()
    embeddings[name] = encoding


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
        unknown = face_encoder(img_in).numpy()
        found = False
        for person in embeddings.keys():
            print(distance(embeddings[person],unknown))
            if distance(embeddings[person],unknown) < 100:
                img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (0, 255, 0), 2)
                img = cv2.putText(img, '{}'.format(person),(int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
                flag = True
                break
        if not found:
            img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (0, 0, 255), 2)
            img = cv2.putText(img, 'Unknown',(int(box[0]), int(box[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))

    cv2.imshow('demo',img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
