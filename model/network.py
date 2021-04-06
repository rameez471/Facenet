import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D,MaxPool2D,PReLU,Permute, Softmax

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

    return model


model = PNet()

with open('../data/p_net_summary.txt','w') as f:
    model.summary(print_fn=lambda x: f.write(x+'\n'))