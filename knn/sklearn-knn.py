# coding=utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import time

def loadImageSet(filename):

    # print "load image set", filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()
 
    head = struct.unpack_from('>IIII', buffers, 0)
    # print "head,", head

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    # print "load imgs finished"
    return imgs

def loadLabelSet(filename):
    # print "load label set", filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    # print "head,", head
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    # print 'load label finished'
    return labels

 
start_time = time.time()
# Load training examples and tesing examples
X_train = loadImageSet('train-images.idx3-ubyte')
X_train = X_train.reshape(60000,784)
y_train = loadLabelSet('train-labels.idx1-ubyte')
y_train = y_train.ravel()
X_test  = loadImageSet('t10k-images.idx3-ubyte')
X_test  = X_test.reshape(10000,784)
y_test  = loadLabelSet('t10k-labels.idx1-ubyte')
 
# Show one of the training example, this part is optional
# i = 0
# pic = X_train[i]
# pic = pic.reshape(28,28,order = 'C')
# plt.imshow(pic,cmap= cm.binary)
# plt.show()
# print 'the label of the picture is',y_train[i]
# Train the model

Model = KNeighborsClassifier(n_neighbors=5)
Model.fit(X_train,y_train) 
# Test the test examples

pred = Model.predict(X_test)
length = len(pred)
count = 0
for i in range(0,length):
    if (pred[i] == y_test[i]):
        count = count + 1
end_time = time.time()
print ('The accuracy is %.2f%%'%(count*1.0*100/length))
print('Spend time：%.3fmin'%((end_time-start_time)/60))
print(classification_report(y_test,pred))