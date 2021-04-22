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


embeddings = []
labels = []

dirname = './lwf/embeddings'
idx = 0
image_dir = './data/test/images'
crop_dirname = './data/test/images_cropped'

crop_size = 160

mtcnn = MTCNN()
face_encoder = tf.keras.models.load_model('./data/facenet_keras.h5')

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
        embedding = face_encoder(img_in)[0].numpy()
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
plt.savefig('PCA.png')

tsne = TSNE(n_components=2, verbose=1,perplexity=40,n_iter=300)
tsne_results = tsne.fit_transform(df[feat_cols].values)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

sns.catplot(
    x='tsne-2d-one', y='tsne-2d-two',
    hue='y',
    palette=sns.color_palette('hls',4),
    data=df,
    legend='full'
)
plt.savefig('t-SNE.png')

pca_10 = PCA(n_components=5)
pca_result = pca_10.fit_transform(df[feat_cols].values)

print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_10.explained_variance_ratio_)))

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result)

df['tsne-pca-10-one'] = tsne_pca_results[:,0]
df['tsne-pca-10-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,10))

sns.catplot(
    x='tsne-pca-10-one', y='tsne-pca-10-one',
    hue='y',
    palette=sns.color_palette('hls',4),
    data=df,
    legend='full'
)
plt.savefig('t-SNE_50.png')