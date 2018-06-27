import time
start = time.time()

from sklearn.datasets import make_swiss_roll
X,t=make_swiss_roll(n_samples=1000,noise=0.2,random_state=42)                  #3D domain,y is a swiss roll
#import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
axes = [-11.5, 14, -2, 23, -12, 15]                                            #x0,x1,y0,y1,z0,z1
fig = plt.figure(figsize=(6, 5))                                               #Plot settings
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()

##PROJECTION
y=t.astype(np.int)                                                             #Logistic regression needs an integer target
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)                      #Kernel trick to reduce the features
X_reduced=rbf_pca.fit_transform(X)
clf=Pipeline([
        ('kpca',KernelPCA(n_components=2)),
        ('log_reg',LogisticRegression())
        ])
param_grid=[{
        'kpca__gamma':np.linspace(0.03,0.05,10),
        'kpca__kernel':['rbf','sigmoid']
        }]
grid_search=GridSearchCV(clf,param_grid,cv=3)
grid_search.fit(X,y)
print(grid_search.best_params_)                                                #Parameters to the best fit

start = time.time()
#LLE Locally Linear Embedding
from sklearn.manifold import LocallyLinearEmbedding
#Measure how each training instance linearly relates to its closest neighbors, and then looking for a low-dimensional representation
#of the trainig set where these local relationships are best preserved
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)                                               #Training
plt.title("Unrolled swiss roll using LLE", fontsize=14)                        #Plot settings
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)
plt.show()

end = time.time()
print(end - start)