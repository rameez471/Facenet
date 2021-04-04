import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Concatenate, BatchNormalization, MaxPool2D, AveragePooling2D, Lambda, Flatten, Dense
from tensorflow.keras.models import Model


def inception_block_1a(x):

    x_3x3 = Conv2D(96, 1, data_format='channels_first')(x)
    x_3x3 = BatchNormalization(axis=1, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)
    x_3x3 = ZeroPadding2D((1,1),data_format='channels_first')(x_3x3)
    x_3x3 = Conv2D(128,3,data_format='channels_first')(x_3x3)
    x_3x3 = BatchNormalization(axis=1, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)

    x_5x5 = Conv2D(16, 1,data_format='channels_first')(x)
    x_5x5 = BatchNormalization(axis=1, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)
    x_5x5 = ZeroPadding2D((2,2), data_format='channels_first')(x_5x5)
    x_5x5 = Conv2D(32,5,data_format='channels_first')(x_5x5)
    x_5x5 = BatchNormalization(axis=1, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)

    x_pool = MaxPool2D(pool_size=(3,3),strides=(2,2),data_format='channels_first')(x)
    x_pool = Conv2D(32,1, data_format='channels_first')(x_pool)
    x_pool = BatchNormalization(axis=1, epsilon=0.00001)(x_pool)
    x_pool = Activation('relu')(x_pool)
    x_pool = ZeroPadding2D(((3,4),(3,4)),data_format='channels_first')(x_pool)

    x_1x1 = Conv2D(64, 1,data_format='channels_first')(x)
    x_1x1 = BatchNormalization(axis=1, epsilon=0.00001)(x_1x1)
    x_1x1 = Activation('relu')(x_1x1)

    inception = Concatenate([x_3x3, x_5x5, x_pool, x_1x1],axis=1)

    return inception

def inception_block_1b(x):

    x_3x3 = Conv2D(96, 1, data_format='channels_first')(x)
    x_3x3 = BatchNormalization(axis=1, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)
    x_3x3 = ZeroPadding2D((1,1),data_format='channels_first')(x_3x3)
    x_3x3 = Conv2D(128,3,data_format='channels_first')(x_3x3)
    x_3x3 = BatchNormalization(axis=1, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)

    x_5x5 = Conv2D(32, 1,data_format='channels_first')(x)
    x_5x5 = BatchNormalization(axis=1, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)
    x_5x5 = ZeroPadding2D((2,2), data_format='channels_first')(x_5x5)
    x_5x5 = Conv2D(64,5,data_format='channels_first')(x_5x5)
    x_5x5 = BatchNormalization(axis=1, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)

    x_pool = MaxPool2D(pool_size=(3,3),strides=(3,3),data_format='channels_first')(x)
    x_pool = Conv2D(64,1, data_format='channels_first')(x_pool)
    x_pool = BatchNormalization(axis=1, epsilon=0.00001)(x_pool)
    x_pool = Activation('relu')(x_pool)
    x_pool = ZeroPadding2D((4,4),data_format='channels_first')(x_pool)

    x_1x1 = Conv2D(64, 1,data_format='channels_first')(x)
    x_1x1 = BatchNormalization(axis=1, epsilon=0.00001)(x_1x1)
    x_1x1 = Activation('relu')(x_1x1)

    inception = Concatenate([x_3x3, x_5x5, x_pool, x_1x1],axis=1)

    return inception

def inception_block_1c(x):

    x_3x3 = Conv2D(128, 1, strides=(1,1), data_format='channels_first')(x)
    x_3x3 = BatchNormalization(axis=1, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)
    x_3x3 = ZeroPadding2D((1,1))(x_3x3)
    x_3x3 = Conv2D(256, 3, strides=(2,2), data_format='channels_first')(x_3x3)
    x_3x3 = BatchNormalization(axis=1, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)

    x_5x5 = Conv2D(32,1,strides=(1,1),data_format='channels_first')(x)
    x_5x5 = BatchNormalization(axis=1, epsilon=0.00001)(x_5)
    x_5x5 = Activation('relu')(x_5x5)
    x_5x5 = ZeroPadding2D((2,2))(x_5x5)
    x_5x5 = Conv2D(64, 5, strides=(2,2),data_format='channels_first')(x_5x5)
    x_5x5 = BatchNormalization(axis=1, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)

    x_pool = MaxPool2D(pool_size=(3,3),strides=(2,2),data_format='channels_first')(x)
    x_pool = ZeroPadding2D(((0,1),(0,1)),data_format='channels_first')(x_pool)

    inception = Concatenate([x_3x3, x_5x5, x_pool], axis=1)

    return inception
