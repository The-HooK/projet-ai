import os
import sys
import random
import warnings

import img as img
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from mahotas import euler
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.color import rgb2gray


def getdata():
    global data_LED5
    global data_LED9
    global data_INC5
    global data_INC9
    data_LED5 = []
    data_LED9 = []
    data_INC5 = []
    data_INC9 = []
    for i in range(1, 9):
        data_LED5.append(rgb2gray(imread('../data/LED/5' + str(i) + '.jpg')))
        data_LED9.append(rgb2gray(imread('../data/LED/9' + str(i) + '.jpg')))
        data_INC5.append(rgb2gray(imread('../data/INC/5' + str(i) + '.jpg')))
        data_INC9.append(rgb2gray(imread('../data/INC/9' + str(i) + '.jpg')))
    # mask = imread()
    global data_ordre0_LED5
    global data_ordre0_LED9
    global data_ordre0_INC5
    global data_ordre0_INC9
    data_ordre0_LED5 = crop(data_LED5)
    data_ordre0_LED9 = crop(data_LED9)
    data_ordre0_INC5 = crop(data_INC5)
    data_ordre0_INC9 = crop(data_INC9)


def crop(image):
    if isinstance(image, list):
        return [img[510:630, 770:890] for img in image]
    else:
        return image[510:630, 770:890]


def calcul_moyenne(image):
    if isinstance(image, list):
        out = np.zeros(len(image))
        for i in range(len(image)):
            out[i] = np.mean(image[i])
        return out
    else:
        return np.mean(image)

def calcul_var(image):
    if isinstance(image, list):
        return [np.var(calcul_projection(img)) for img in image]
    else:
        return np.var(calcul_projection(image))

def calcul_cov(image):
    if isinstance(image, list):
        return [np.cov(calcul_projection(img), calcul_projection(img.transpose())) for img in image]
    else:
        return np.cov(calcul_projection(image), calcul_projection(image.transpose()))


def calcul_conv(image):
    if isinstance(image, list):
        return [sum(calcul_projection(img) * calcul_projection(img.transpose())) for img in image]
    else:
        return sum(calcul_projection(image) * calcul_projection(image.transpose()))


def calcul_projection(image):
    if isinstance(image, list):
        return [sum(img) for img in image]
    else:
        return sum(image)


def plot_feature(func):
    if isinstance(func, list):
        if len(func) > 3:
            print("i can only plot features in 3D space max, sorry")
            return
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(func[0](data_ordre0_LED5), func[1](data_ordre0_LED5), func[2](data_ordre0_LED5), c='blue')
        ax.scatter(func[0](data_ordre0_LED9), func[1](data_ordre0_LED9), func[2](data_ordre0_LED9), c='red')
        ax.scatter(func[0](data_ordre0_INC5), func[1](data_ordre0_INC5), func[2](data_ordre0_INC5), c='green')
        ax.scatter(func[0](data_ordre0_INC9), func[1](data_ordre0_INC9), func[2](data_ordre0_INC9), c='orange')
        plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])
        ax.set_xlabel(func[0].__name__)
        ax.set_ylabel(func[1].__name__)
        ax.set_zlabel(func[2].__name__)
        plt.show()
    else:
        plt.figure()
        plt.scatter(range(len(data_ordre0_LED5)), func(data_ordre0_LED5), c='blue')
        plt.scatter(range(len(data_ordre0_LED9)), func(data_ordre0_LED9), c='red')
        plt.scatter(range(len(data_ordre0_INC5)), func(data_ordre0_INC5), c='green')
        plt.scatter(range(len(data_ordre0_INC9)), func(data_ordre0_INC9), c='orange')
        plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])
        plt.xlabel('data')
        plt.ylabel(func.__name__)
        plt.show()

getdata()
"""
plt.figure()
plt.subplot(221)
imshow(data_ordre0_LED5[1])
plt.subplot(222)
imshow(data_ordre0_LED9[1])
plt.subplot(223)
imshow(data_ordre0_INC5[1])
plt.subplot(224)
imshow(data_ordre0_INC9[1])
plt.show()
"""
plot_feature(calcul_moyenne)
plot_feature(calcul_var)
plot_feature(calcul_conv)

plot_feature([calcul_moyenne, calcul_var, calcul_conv])

## plot_feature(calcul_cov)
#
#noms = ["\delta_x
for i in [0, 1, 3]:
    plt.figure()
    plt.scatter(range(len(data_ordre0_LED5)), [calcul_cov(img).reshape(4)[i] for img in data_ordre0_LED5], c='blue')
    plt.scatter(range(len(data_ordre0_LED9)), [calcul_cov(img).reshape(4)[i] for img in data_ordre0_LED9], c='red')
    plt.scatter(range(len(data_ordre0_INC5)), [calcul_cov(img).reshape(4)[i] for img in data_ordre0_INC5], c='green')
    plt.scatter(range(len(data_ordre0_INC9)), [calcul_cov(img).reshape(4)[i] for img in data_ordre0_INC9], c='orange')
    plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])
    plt.axis()


plt.figure()
plt.subplot(211)
plt.plot(range(120), calcul_projection(np.mean(data_ordre0_LED5, 0)), c='blue')
plt.plot(range(120), calcul_projection(np.mean(data_ordre0_LED9, 0)), c='red')
plt.plot(range(120), calcul_projection(np.mean(data_ordre0_INC5, 0)), c='green')
plt.plot(range(120), calcul_projection(np.mean(data_ordre0_INC9, 0)), c='orange')
for i in range(8):
    plt.plot(range(120), calcul_projection(data_ordre0_LED5[i]), '--', c='blue', alpha=0.3)
    plt.plot(range(120), calcul_projection(data_ordre0_LED9[i]), '--', c='red', alpha=0.3)
    plt.plot(range(120), calcul_projection(data_ordre0_INC5[i]), '--', c='green', alpha=0.3)
    plt.plot(range(120), calcul_projection(data_ordre0_INC9[i]), '--', c='orange', alpha=0.3)
plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])

plt.subplot(212)
plt.plot(range(120), calcul_projection(np.mean([el.transpose() for el in data_ordre0_LED5], 0)), c='blue')
plt.plot(range(120), calcul_projection(np.mean([el.transpose() for el in data_ordre0_LED9], 0)), c='red')
plt.plot(range(120), calcul_projection(np.mean([el.transpose() for el in data_ordre0_INC5], 0)), c='green')
plt.plot(range(120), calcul_projection(np.mean([el.transpose() for el in data_ordre0_INC9], 0)), c='orange')
for i in range(8):
    plt.plot(range(120), calcul_projection(data_ordre0_LED5[i].transpose()), '--', c='blue', alpha=0.3)
    plt.plot(range(120), calcul_projection(data_ordre0_LED9[i].transpose()), '--', c='red', alpha=0.3)
    plt.plot(range(120), calcul_projection(data_ordre0_INC5[i].transpose()), '--', c='green', alpha=0.3)
    plt.plot(range(120), calcul_projection(data_ordre0_INC9[i].transpose()), '--', c='orange', alpha=0.3)
    print(i)
plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])
