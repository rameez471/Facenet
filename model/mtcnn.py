import tensorflow as tensorflow
from .network import *
from .utils import * 

DEFAULT_THESHOLDSS = [0.7,0.8,0.9]
DEFAULT_NMS_THRESHOLDS = [0.6,0.6,0.6]

class MTCNN(object):

    def __init__(self,min_face_size=20.0,thresholds=None,
                nms_threshold=None,max_output_size=300):
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.min_face_size = min_face_size
        self.thresholds = thresholds or DEFAULT_THESHOLDSS
        self.nms_threshold = nms_threshold or DEFAULT_NMS_THRESHOLDS
        self.max_output_size = max_output_size
        self.scale_cache = {}

    def get_scales(self,height,width):
        
        min_length = min(height,width)
        if min_length in self.scale_cache:
            return self.scale_cache[min_length]

        min_detection_size = 12
        factor = 0.707
        scales = []
        m = min_detection_size / self.min_face_size
        min_length *= m
        factor_count = 0

        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        self.scale_cache[min_length] = scales

        return scales

    @tf.function()
    def stage_one_scale(self, img, height, width, scale):
        
        hs = tf.math.ceil(height * scale)
        ws = tf.math.ceil(width * scale)

        img_in = tf.image.resize(img,(hs,ws))
        img_in = preprocess(img_in)
        img_in = tf.expand_dims(img_in, 0)

        probs, offsets = self.pnet(img_in)
        boxes = generate_boxes(probs[0], offsets[0], scale, self.thresholds[0])

        if len(boxes) == 0:
            return boxes

        keep = tf.image.non_max_suppression(boxes[:,0:4], boxes[:,4], self.max_output_size, iou_threshold=0.5)
        boxes = tf.gather(boxes,keep)

        return boxes


    @tf.function()
    def stage_one_filter(self,boxes):

        bboxes,score,offsets = boxes[:,:4], boxes[:,4], boxes[:,5:]

        bboxes = callibrate_bbox(bboxes,offsets)
        bboxes = box_to_square(bboxes)

        keep = tf.image.non_max_suppression(bboxes, score, self.max_output_size, iou_threshold=self.nms_threshold[0])

        bboxes = tf.gather(bboxes, keep)

        return bboxes
    
    def stage_one(self, img, scale):

        height, width, _ = img.shape

        boxes = []
        for s in scale:
            boxes.append(self.stage_one_scale(img,height, width, s))
        boxes = tf.concat(boxes, 0)

        if boxes.shape[0] == 0:
            return []

        return self.stage_one_filter(boxes)

    def stage_two(self, bboxes, img, height, width, num_boxes):

        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes)

        probs, offsets = self.rnet(img_boxes)

        keep = tf.where(probs[:,1] > self.thresholds[1])[:,0]
        bboxes = tf.gather(bboxes, keep)
        scores = tf.gather(probs[:,1], keep)
        offsets = tf.gather(offsets, keep)

        bboxes = callibrate_bbox(bboxes, offsets)
        bboxes = box_to_square(bboxes)

        keep = tf.image.non_max_suppression(bboxes, scores, self.max_output_size, self.nms_threshold[1])
        bboxes = tf.gather(bboxes, keep)

        return bboxes

    @tf.function()
    def stage_three(self, bboxes, img, height, width, num_boxes):

        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size= 48)
        probs, offsets, landmarks = self.onet(img_boxes)

        keep = tf.where(probs[:,1] > self.thresholds[2])[:,0]
        bboxes = tf.gather(bboxes, keep)
        offsets = tf.gather(offsets, keep)
        scores = tf.gather(probs[:,1], keep)
        landmarks = tf.gather(landmarks, keep)

        width = tf.expand_dims(bboxes[:,2] - bboxes[:,0]+1.0, 1)
        height = tf.expand_dims(bboxes[:,3] - bboxes[:,1]+1.0, 1)
        xmin = tf.expand_dims(bboxes[:,0],1)
        ymin = tf.expand_dims(bboxes[:,1],1)
        landmarks = tf.concat([xmin + width * landmarks[:,0:5],
                              ymin + height * landmarks[:,5:10]],1)
        
        bboxes = callibrate_bbox(bboxes, offsets)
        keep = tf.image.non_max_suppression(bboxes, scores, self.max_output_size, self.nms_threshold[2])

        bboxes = tf.gather(bboxes, keep)
        landmarks = tf.gather(landmarks, keep)
        scores = tf.gather(scores, keep)

        return bboxes, landmarks, scores


    def detect(self, img):

        height, width, _ = img.shape

        img = tf.convert_to_tensor(img, tf.float32)
        scales = self.get_scales(height, width)

        bboxes = self.stage_one(img, scales)
        if len(bboxes) == 0:
            return [],[],[]

        bboxes = self.stage_two(bboxes, img, height, width, bboxes.shape[0])
        if len(bboxes) == 0:
            return [],[],[]

        bboxes, landmarks, scores = self.stage_three(bboxes, img, height, width, bboxes.shape[0])

        return bboxes, landmarks, scores

