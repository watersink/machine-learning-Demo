import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets


#使用pca的主成分特征，得到旋转矩阵，将数据旋转正

iris = datasets.load_iris()
X = iris.data[:50,:2]



pca = decomposition.PCA(n_components=2)
pca.fit(X)



# number of components
n_pcs= pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
print("most_important",most_important)
print("pca.components_",pca.components_)
if most_important!=list(range(len(most_important))):
    reordered = np.vstack((pca.components_[most_important[0], :],pca.components_[most_important[1], :]))
else:
    reordered = pca.components_


out_components = np.matmul(X, np.transpose(reordered))



fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,0], X[:,1],c='b', marker='o')
ax.scatter(out_components[:,0], out_components[:,1],c='r', marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()

