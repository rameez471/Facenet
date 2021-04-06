import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D,MaxPool2D,PReLU,Permute, Softmax, Flatten, Dense

def PNet():

    x_input = Input((None,None,3))

    x = Permute((2,1,3), name='pnet_permute_1')(x_input)
    x = Conv2D(10,3,1,name='pnet_conv_1')(x)
    x = PReLU(shared_axes=[1,2],name='prelu_1')(x)
    x = MaxPool2D(2,2,padding='same', name='pnet_maxpool_1')(x)

    x = Conv2D(16,3,1,name='pnet_conv_2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu_2')(x)

    x = Conv2D(32,3,1, name='pnet_conv_3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu_3')(x)

    #Branch 1
    a = Conv2D(2,1,1,name='pnet_branch_a')(x)
    a = Softmax()(a)
    a = tf.transpose(a, perm=[0,2,1,3])

    #Branch 2
    b = Conv2D(4,1,1,name='pnet_branch_b')(x)
    b = tf.transpose(b, perm=[0,2,1,3])

    model = Model(inputs=x_input, outputs=[a,b])

    model.load_weights('../data/pnet.h5')

    return model


def RNet():

    x_input = Input((24,24,3))

    x = Permute((2,1,3),name='rnet_permute_1')(x_input)
    x = Conv2D(28,3,1,name='rnet_conv_1')(x)
    x = PReLU(shared_axes=[1,2],name='rnet_prelu_1')(x)
    x = MaxPool2D(3,2,padding='same',name='rnet_maxpool_1')(x)

    x = Conv2D(48,3,1,name='rnet_conv_2')(x)
    x = PReLU(shared_axes=[1,2],name='rnet_prelu_2')(x)
    x = MaxPool2D(3,2,name='rnet_maxpool_2')(x)

    x = Conv2D(64,2,1,name='rnet_conv_3')(x)
    x = PReLU(shared_axes=[1,2],name='rnet_prelu_3')(x)

    x = Flatten(name='rnet_flatten')(x)
    x = Dense(128, name='rnet_dense')(x)
    x = PReLU(name='rnet_prelu_4')(x)

    #Branch 1
    a = Dense(2, name='rent_branch_1')(x)
    a = Softmax()(a)

    #Branch 2
    b = Dense(4,name='rnet_brnach_2')(x)

    model = Model(inputs=x_input, outputs=[a,b])

    model = model.load_weights('../data/rnet.h5')

    return model


def ONet():

    x_input = Input((48,48,3))

    x = Permute((2,1,3),name='onet_permute')(x_input)
    x = Conv2D(32,3,1,name='onet_conv_1')(x)
    x = PReLU(shared_axes=[1,2],name='onet_prelu_1')(x)
    x = MaxPool2D(3,2,padding='same',name='onet_maxpool_1')(x)

    x = Conv2D(64,3,1,name='onet_conv_2')(x)
    x = PReLU(shared_axes=[1,2],name='rnet_prelu_2')(x)
    x = MaxPool2D(3,2,name='onet_maxpool_2')(x)

    x = Conv2D(64,3,1,name='onet_conv_3')(x)
    x = PReLU(shared_axes=[1,2],name='onet_prelu_3')(x)
    x = MaxPool2D(2,2,padding='same',name='onet_maxpool_3')(x)

    x = Conv2D(128,2,1,name='onet_conv_4')(x)
    x = PReLU(shared_axes=[1,2],name='onet_prelu_4')(x)

    x = Flatten(name='onet_flatten')(x)
    x = Dense(256,name='oenet_dense')(x)
    x = PReLU(name='onet_prelu_5')(x)

    # Output 1 
    a = Dense(2, name='output_1')(x)
    a = Softmax()(a)

    #Output 2
    b = Dense(4, name='output_2')(x)

    #Output 3
    c = Dense(10, name='output_3')(x)

    model = Model(inputs=x_input, outputs=[a,b,c])

    model.load_weights('../data/onet.h5')

    return model


# pnet = PNet()
# rnet = RNet()
# onet = ONet()

# print('I did it bitch!!')