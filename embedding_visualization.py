import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
import cv2
from model.mtcnn import MTCNN
import tensorflow as tf
import pandas as pd
from model.inception import InceptionResNetV2
from mpl_toolkits.mplot3d import Axes3D

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x,x)))

embeddings = []
labels = []

dirname = './lwf/embeddings'
idx = 0
image_dir = './data/test/images'
crop_dirname = './data/test/images_cropped'

crop_size = 160

mtcnn = MTCNN()
face_encoder = InceptionResNetV2()
face_encoder.load_weights('./data/facenet_keras_weights.h5')

embeddings,labels = [],[]
idx = 0
class_name = []

for folder in os.listdir(crop_dirname):
    class_name.append(folder)
    for file in os.listdir(crop_dirname+'/'+folder):
        img = cv2.imread(crop_dirname + '/'+folder + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.image.resize(img,(160,160))
        img_in *= 1./255
        img_in = tf.expand_dims(img_in,0)
        embedding = face_encoder(img_in).numpy()
        embedding = l2_normalize(embedding)[0]
        embeddings.append(embedding)
        labels.append(idx)
    idx += 1


print('Total Embeddings: ', len(embeddings))
print('Total Labels: ', len(labels))

embeddings = np.array(embeddings)
labels = np.array(labels)

print(embeddings.shape)
print(labels.shape)


feat_cols = [ 'embedding_'+str(i) for i in range(embeddings.shape[1]) ]

df = pd.DataFrame(embeddings, columns=feat_cols)
df['y'] = labels
df['labels'] = df['y'].apply(lambda i: str(i))

X, y = None, None

print('Size pf the dataframe: ',df.shape)
np.random.seed(42)

rndperm = np.random.permutation(df.shape[0])


pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca_one'] = pca_result[:,0]
df['pca_two'] = pca_result[:,1]
df['pca_three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


plt.figure(figsize=(16,10))
sns.scatterplot(
    x='pca_one', y='pca_two',
    hue='y',
    palette = sns.color_palette('hls',4),
    data=df.loc[rndperm,:],
    legend='full'
)
plt.savefig('./data/test/PCA_pretrained.png')

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs = df.loc[rndperm,:]['pca_one'],
    ys = df.loc[rndperm,:]['pca_two'],
    zs = df.loc[rndperm,:]['pca_three'],
    c = df.loc[rndperm,:]['y'],
)
ax.set_xlabel('pca_one')
ax.set_ylabel('pca_two')
ax.set_zlabel('pca_three')
plt.savefig('./data/test/3D_pretrained.png')


