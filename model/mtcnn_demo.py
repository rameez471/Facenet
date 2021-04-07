import cv2
from mtcnn import MTCNN

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


mtcnn = MTCNN()
# img = cv2.imread('wp4013886.jpg')
# img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# bboxes, landmarks, scores = mtcnn.detect(img_in)

# img = draw_faces(img, bboxes, landmarks, scores)
# cv2.imshow('result',img)
# cv2.imwrite('result.jpg', img)

video_path = 'michael.mp4'
output_path = 'michael_result.mp4'

vid = cv2.VideoCapture(video_path)
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