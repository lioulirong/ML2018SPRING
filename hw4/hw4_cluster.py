import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
import pandas as pd
import csv
import sys

filename = 'image.npy'
images = np.load(sys.argv[1]);

data = []

for row in images:
    data.append(row)

x = np.array(data,np.float64)
#plt.imshow(x[0].reshape(28,28).astype(np.uint8))
pca = PCA(n_components=400,whiten=True,svd_solver='full')
x_new = pca.fit_transform(x)

kmeans = KMeans(n_clusters=2).fit_predict(x_new)

df=pd.read_csv(sys.argv[2])
y=np.array(df)

f = open(sys.argv[3],"w")
w = csv.writer(f)
w.writerow(('ID','Ans'))
for i in range(len(y)):
    if(kmeans[y[i,1]]==kmeans[y[i,2]]):
        d=1
    else:
        d=0
    w.writerow((i,d))
f.close()