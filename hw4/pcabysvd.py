
from skimage import io
from skimage import transform as tf
import numpy as np
import os
import sys

resize = 600

fimgs = []
for i in range(0,415):
    filename = os.path.join( sys.argv[1],str(i) ) + '.jpg'
    img = io.imread(filename)
    img = tf.resize(img,(resize,resize,3),preserve_range=True)
    fimgs.append(img)

#print('milestone')

X = np.array(fimgs)#.astype(np.float64)
X_mean = np.mean(X,axis=0).astype(np.float64)
#io.imsave("mean_face.jpg",X_mean.astype(np.uint8))
X -= X_mean
X = X.reshape(415,-1);
X = X.T
u,s,v = np.linalg.svd(X,full_matrices= False)

U = u.T

# eigenface = np.zeros((415,resize*resize*3))
# for i in range(415):
#     eigenface[i] = U[i] - U[i].min()
#     eigenface[i] / eigenface[i].max()
#     eigenface[i] *= 255

filename = os.path.join( sys.argv[2] )

img = io.imread(filename)
img = tf.resize(img,(resize,resize,3),preserve_range=True)
img -= X_mean
img = img.reshape(-1,1)
Cs = np.dot(U[:4,:],img)
recon = np.dot(Cs.T,U[:4,:])
recon = recon.reshape(resize,resize,3).astype(np.uint8) + X_mean.astype(np.uint8)
io.imsave("reconstruction.jpg",recon)



