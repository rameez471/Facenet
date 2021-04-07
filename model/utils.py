import tensorflow as tf

def box_to_square(bbox):
    """
    Converting bounding box into square
    Input: Bounding box shape = (n,4)

    Output: Square bounding box = (n,4)
    """
    x1, y1, x2, y2 = [bbox[:,i] for i in range(4)]

    w = x2- x1
    h = y2- y1

    m = tf.maximum(h,w)

    dx1 = x1 + 0.5*w - m*0.5
    dy1 = y1 + 0.5*h - m*0.5
    dx2 = dx1 + m
    dy2 = dy1 + m

    return tf.stack([
        tf.math.round(dx1),
        tf.math.round(dy1),
        tf.math.round(dx2),
        tf.math.round(dy2),
    ],1)


def callibrate_bbox(bboxes, offsets):
    """
    Correcting bounding box by offsets given by network
    Input: 
        bbox: bounding box of shape (n,4)
        offsets: offsets returned by network of shape (n,4)
    Output:
        Coorrected bounding box of shape (n,4)
    """
    x1, y1, x2, y2 = [bboxes[:,i] for i in range(4)]

    w = x2 - x1
    h = y2 - y1

    correction = tf.stack([w,h,w,h],1) * offsets

    return bboxes + correction

def preprocess(img):
    """
    Preprocess image before feeding network
    """
    img = (img - 127.5) * 0.0078125
    return img

def get_image_boxes(bboxes, img, height, width, num_boxes, size=24):
    """
    Cut out boxes from the images
    Input:
        bboxes: Bounding boxes
        img: Image tensor
        height: Height of the image
        width: Width of the image
        num_boxes: Number of boxes
        size: Size of the cut-out
    """
    x1 = tf.math.maximum(bboxes[:,0],0.0) / width
    y1 = tf.math.maximum(bboxes[:,1],0.0) / height
    x2 = tf.math.minimum(bboxes[:,2],width) / width
    y2 = tf.math.minimum(bboxes[:,3],height) / height

    boxes = tf.stack([y1,x1,y2,x2],1)
    img_boxes = tf.image.crop_and_resize(tf.expand_dims(img,0),boxes,tf.zeros(num_boxes, dtype=tf.int32),(size,size))

    img_boxes = preprocess(img_boxes)

    return img_boxes

def generate_boxes(probs, offsets, scale, threshold):

    stride = 2
    cell_size = 12

    probs = probs[:,:,1]

    indices = tf.where(probs > threshold)
    if indices.shape[0] == 0:
        return tf.zeros((0,9))

    offsets = tf.gather_nd(offsets, indices)
    scores = tf.expand_dims(tf.gather_nd(probs, indices), axis=1)

    indices = tf.cast(indices,tf.float32)

    bboxes = tf.concat([
        tf.expand_dims(tf.math.round((stride * indices[:, 1]) / scale), 1),
        tf.expand_dims(tf.math.round((stride * indices[:, 0]) / scale), 1),
        tf.expand_dims(tf.math.round((stride * indices[:, 1] + cell_size) / scale), 1),
        tf.expand_dims(tf.math.round((stride * indices[:, 0] + cell_size) / scale), 1),
        scores, offsets
    ], 1)

    return bboxes