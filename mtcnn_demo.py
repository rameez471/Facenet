import cv2
from model.mtcnn import MTCNN
import argparse
import tensorflow as tf
import string
import random

def draw_faces(img, bboxes, landmarks, scores):
    for box, landmark, score in zip(bboxes, landmarks, scores):
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
        for i in range(5):
            x = int(landmark[i])
            y = int(landmark[i + 5])
            img = cv2.circle(img, (x, y), 1, (0, 255, 0))
        img = cv2.putText(img, '{:.2f}'.format(score), (int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
        
            
    return img

def image(input_path):

    mtcnn = MTCNN()
    img = cv2.imread(input_path)
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes, landmarks, scores = mtcnn.detect(img_in)

    img = draw_faces(img, bboxes, landmarks, scores)
    cv2.imshow('result',img)
    cv2.imwrite(input_path+'result.jpg', img)


def video(input_path, output_path):

    mtcnn = MTCNN()

    vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))


    print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)


    while True:
        _, img = vid.read()
        if img is None:
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes, landmarks, scores = mtcnn.detect(img_in)
        img = draw_faces(img, bboxes, landmarks, scores)

        cv2.imshow('demo',img)
        if output_path is not None:
            out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cv2.destroyAllWindows()


input_path = './examples/michael.mp4'
output_path = './examples/rec_result.mp4'


video(input_path, output_path)

# parser = argparse.ArgumentParser(description='Demo')

# parser.add_argument('--image_input',help='Inpt Image')

# parser.add_argument('--video_input',help='Input Video')
# parser.add_argument('--video_output',help='Output video location')

# if __name__ == '__main__':

#     args = parser.parse_args()
#     if args.image_input:
#         image(args.image_input)

#     else:
#         video(args.video_input, args.video_output)
