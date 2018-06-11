
# -*- coding: utf-8 -*- 
# knn.ipynb笔记
# Box1
# Run some setup code for this notebook.
# 运行一些设置代码

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

from future import print_function

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
# 下面这行代码 设置matplotlib画图在notebook页面中显示
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots 设置plt的大小
plt.rcParams['image.interpolation'] = 'nearest'#最邻近插值
plt.rcParams['image.cmap'] = 'gray'#显示灰度图像

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2


#Box2
# Load the raw CIFAR-10 data.
#加载CIFAR-10数据
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
# 输出 训练数据和测试数据的大小（sanity check 完整性检查）
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

#Box3
# Visualize some examples from the dataset.
# 虚拟化一些数据集里的例子
# We show a few examples of training images from each class.
# 展示部分训练数据 每类选几张
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 计算类型的总数
num_classes = len(classes)
# 设置每类选几张图片
samples_per_class = 7
for y, cls in enumerate(classes):
    # 查找非零（返回值是非零值的index下标）
    idxs = np.flatnonzero(y_train == y)
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # a为array或者array(int),size是返回的格式（m*n*p），replace表示是否可重默认ture可重，p表示概率
    # >>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
    # array([3, 0, 2])
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
# enumerate将其组成一个索引序列，利用它可以同时获得索引和值(索引，值)
for i, idx in enumerate(idxs):
    plt_idx = i * num_classes + y + 1
    #subplot(numRows, numCols, plotNum)行, 列, 画图位置 
    plt.subplot(samples_per_class, num_classes, plt_idx)
    plt.imshow(X_train[idx].astype('uint8'))
    plt.axis('off')
    if i == 0:
        plt.title(cls)
plt.show()

#Box4
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Box5
# Reshape the image data into rows
# reshape(array,newshape)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))