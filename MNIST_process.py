import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# image_train, lable_train = load_mnist(r'C:\Users\HUAWEI\Desktop\dataset\MNIST\raw',kind='train')
# print(type(image_train), image_train.shape)
# print(type(lable_train), lable_train.shape)
#
# image_test, lable_test = load_mnist(r'C:\Users\HUAWEI\Desktop\dataset\MNIST\raw',kind='test')
# print(type(image_test), image_test.shape)
# print(type(lable_test), lable_test.shape)
#
# image = np.vstack([image_train, image_test])
# image = image / 255.0
#
# lable = np.hstack([lable_train, lable_test]).reshape(-1, 1)
# print(type(image), image.shape)
# print(type(lable), lable.shape)
#
# data = np.hstack([image, lable]).astype(np.float)
# print(type(data), data.shape)
#
#
# np.savetxt('./MNIST.data', data, fmt='%f', delimiter=',')

data = np.loadtxt('./MNIST.data', delimiter=',')
print(np.sum(data[0]))

