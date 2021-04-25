import tensorflow as tf
import numpy as np
import os
import cv2
from model.inception import InceptionResNetV2
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import time

def buildDataset(dirname):
    X,y = [],[]
    idx = 0
    print(dirname)
    for folder in os.listdir(dirname):
        for file in os.listdir(dirname+'/'+folder):
            img = cv2.imread(dirname+'/'+folder+'/'+file)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(idx)
        idx+=1

    X = np.array(X)
    y = np.array(y)

    dataset = []

    for n in range(idx):
        images_class_n = np.asarray([row for idx,row in enumerate(X) if y[idx] == n])
        dataset.append(images_class_n/255)

    return dataset, X, y, idx

def get_random(batch_size):
    m,w,h,c = X[0].shape

    triplets = [np.zeros((batch_size,h,w,c)) for i in range(3)]

    for i in range(batch_size):
        
        anchor_class = np.random.randint(0,nb_classes)
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]

        [idx_A, idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)

        negative_class = (anchor_class + np.random.randint(1,nb_classes)) % nb_classes
        nb_sample_available_for_class_N = X[negative_class].shape[0]

        idx_N = np.random.randint(0,nb_sample_available_for_class_N)

        triplets[0][i,:,:,:] = X[anchor_class][idx_A,:,:,:]
        triplets[1][i,:,:,:] = X[anchor_class][idx_P,:,:,:]
        triplets[2][i,:,:,:] = X[negative_class][idx_N,:,:,:]

    return triplets

def drawTriplets(triplebatch, nbmax=None):

    labels = ['Anchor','Positives','Negative']

    if nbmax==None:
        nbrows = triplebatch[0].shape[0]
    else:
        nbrows = min(nbmax,triplebatch[0].shape[0])

    for row in range(nbrows):
        fig = plt.figure(figsize=(16,2))

        for i in range(3):
            subplot = fig.add_subplot(1,3,i+1)
            plt.axis('off')
            plt.imshow(triplebatch[i][row,:,:,0],vmin=0,vmax=1)
            subplot.title.set_text(labels[i])
        plt.show()


def compute_dist(a,b):
    return np.sum(np.square(a-b))

def get_batch_hard(draw_batch_size, hard_batch_size, norm_batchs_size,network):
    m,w,h,x = X[0].shape

    studybatch = get_random(draw_batch_size)
    
    studybatchloss = np.zeros((draw_batch_size))

    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])

    studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)

    selection = np.argsort(studybatchloss)[::-1][:hard_batch_size]

    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)

    selection = np.append(selection,selection2)

    triplets = [studybatch[0][selection,:,:,:], studybatch[1][selection,:,:,:],studybatch[2][selection,:,:,:]]

    return triplets

class TripletLossLayer(Layer):

    def __init__(self,alpha,**kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positve, negative = inputs
        p_dist = K.sum(K.square(anchor-positve),axis=-1)
        n_dist = K.sum(K.square(anchor-negative),axis=-1)

        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0),axis=0)

    def call(self,inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def built_model(input_shape, network, margin=0.2):

    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])

    network_train = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)

    return network_train



dirname = './data/test/images_cropped'
dataset, X, y,nb_classes = buildDataset(dirname)
dataset = np.array(dataset)
X = dataset
face_encoder = InceptionResNetV2()
# face_encoder.load_weights('./data/test/bvs.h5')


model = built_model((160,160,3),face_encoder)
optimizer = tf.keras.optimizers.Adam(lr=0.00006)
model.compile(loss=None, optimizer=optimizer)

print('Starting training.....')

t_start = time.time()
n_iter = 20

for i in range(1,n_iter+1):
    triplets = get_batch_hard(128,16,16,face_encoder)
    loss = model.train_on_batch(triplets,None)
    
    print('\n------------\n')
    print(" Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss))

    i+=1

face_encoder.save_weights('./data/test/bvs_new.h5')