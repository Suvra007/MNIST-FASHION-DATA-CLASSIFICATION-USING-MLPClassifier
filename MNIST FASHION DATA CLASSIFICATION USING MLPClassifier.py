import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
imagefile="mnist\\t10k-images-idx3-ubyte"
Xt=idx2numpy.convert_from_file(imagefile)
Xt.resize(10000,784)
labelfile="mnist\\t10k-labels-idx1-ubyte"
yt=idx2numpy.convert_from_file(labelfile)
fig,ax=plt.subplots(5,5)
ax=ax.flatten()
for i in range(25):
    img=Xt[yt==0][i].reshape(28,28)
    ax[i].imshow(img,cmap="Greys",interpolation="nearest")
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
imagefile="mnist\\train-images-idx3-ubyte"
X=idx2numpy.convert_from_file(imagefile)
X.resize(60000,784)
labelfile="mnist\\train-labels-idx1-ubyte"
y=idx2numpy.convert_from_file(labelfile)
fig,ax=plt.subplots(5,5)
ax=ax.flatten()
for i in range(25):
    img=X[y==0][i].reshape(28,28)
    ax[i].imshow(img,cmap="Greys",interpolation="nearest")
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
clf=MLPClassifier(hidden_layer_sizes=(50,),batch_size=50,activation="logistic",alpha=0.001,shuffle=True,solver="adam",learning_rate="constant",learning_rate_init=0.001).fit(X,y)
yp=clf.predict(Xt)
print(classification_report(yt,yp))
d=plot_confusion_matrix(clf,Xt,yt)
d.figure_.suptitle("Confusion matrix")
print("Confusion matrix:\n%s"%d.confusion_matrix)
misc_img=Xt[yt!=yp][:10]
correct_lab=yt[yt!=yp][:10]
misc_lab=yp[yt!=yp][:10]
fig,ax=plt.subplots(2,5)
ax=ax.flatten()
for i in range(10):
    img=misc_img[i].reshape(28,28)
    ax[i].imshow(img,cmap="Greys",interpolation="nearest")
    ax[i].set_title("%d) t: %d p: %d" % (i+1,correct_lab[i],misc_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
c_img=Xt[yt==yp][:10]
correct_lab=yt[yt==yp][:10]
c_lab=yp[yt==yp][:10]
fig,ax=plt.subplots(2,5)
ax=ax.flatten()
for i in range(10):
    img=c_img[i].reshape(28,28)
    ax[i].imshow(img,cmap="Greys",interpolation="nearest")
    ax[i].set_title("%d) t: %d p: %d" % (i+1,correct_lab[i],c_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
