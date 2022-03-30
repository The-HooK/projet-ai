import os
import sys
import random
import warnings

import img as img
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier

from mahotas import euler
from skimage.io import imread, imread_collection, concatenate_images
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
        data_LED5.append(rgb2gray(imread('../data/LED_cropped_bin/5' + str(i) + '.jpg')))
        data_LED9.append(rgb2gray(imread('../data/LED_cropped_bin/9' + str(i) + '.jpg')))
        data_INC5.append(rgb2gray(imread('../data/INC_cropped_bin/5' + str(i) + '.jpg')))
        data_INC9.append(rgb2gray(imread('../data/INC_cropped_bin/9' + str(i) + '.jpg')))
    # mask = imread()
    global data_ordre0_LED5
    global data_ordre0_LED9
    global data_ordre0_INC5
    global data_ordre0_INC9
    # data_ordre0_LED5 = crop(data_LED5)
    # data_ordre0_LED9 = crop(data_LED9)
    # data_ordre0_INC5 = crop(data_INC5)
    # data_ordre0_INC9 = crop(data_INC9)

    data_ordre0_LED5 = data_LED5
    data_ordre0_LED9 = data_LED9
    data_ordre0_INC5 = [data[:66, :66] for data in data_INC5]
    data_ordre0_INC9 = [data[:66, :66] for data in data_INC9]

    return data_ordre0_LED5 + data_ordre0_LED9 + data_ordre0_INC5 + data_ordre0_INC9


def crop(image):
    if isinstance(image, list):
        return [img[510:630, 770:890] for img in image]
    else:
        return image[510:630, 770:890]


def calcul_moyenne(image):
    if isinstance(image, list):
        return [np.mean(img) for img in image]
    else:
        return np.mean(image)


def calcul_var(image):
    if isinstance(image, list):
        return [np.var(calcul_projection(img)) for img in image]
    else:
        return np.var(calcul_projection(image))


def calcul_cov(image):
    if isinstance(image, list):
        return [np.cov(calcul_projection(img), calcul_projection(img.T))[1, 0] for img in image]
    else:
        return np.cov(calcul_projection(image), calcul_projection(image.T))[1, 0]


def calcul_conv(image):
    if isinstance(image, list):
        return [sum(calcul_projection(img) * calcul_projection(img.T)) for img in image]
    else:
        return sum(calcul_projection(image) * calcul_projection(image.T))


def calcul_projection(image):
    if isinstance(image, list):
        return [sum(img) for img in image]
    else:
        return sum(image)


def plot_images():
    # album_LED5 = np.concatenate(np.array([data_ordre0_LED5]).reshape((4,2)))
    names = ['LED5', 'LED9', 'INC5', 'INC9']
    album = [
        np.concatenate([np.concatenate(data_ordre0_LED5[:4], axis=1), np.concatenate(data_ordre0_LED5[4:], axis=1)]),
        np.concatenate([np.concatenate(data_ordre0_LED9[:4], axis=1), np.concatenate(data_ordre0_LED9[4:], axis=1)]),
        np.concatenate([np.concatenate(data_ordre0_INC5[:4], axis=1), np.concatenate(data_ordre0_INC5[4:], axis=1)]),
        np.concatenate([np.concatenate(data_ordre0_INC9[:4], axis=1), np.concatenate(data_ordre0_INC9[4:], axis=1)])]

    fig = plt.figure()
    for i in range(4):
        plt.subplot(221 + i)
        plt.imshow(album[i])

        frame1 = fig.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.gray()
        plt.title(names[i])
    plt.show()


def plot_proj():
    plt.figure()
    plt.subplot(211)
    plt.plot(range(len(data_ordre0_LED5[1][1])), calcul_projection(np.mean(data_ordre0_LED5, 0)), c='blue')
    plt.plot(range(len(data_ordre0_LED9[1][1])), calcul_projection(np.mean(data_ordre0_LED9, 0)), c='red')
    plt.plot(range(len(data_ordre0_INC5[1][1])), calcul_projection(np.mean(data_ordre0_INC5, 0)), c='green')
    plt.plot(range(len(data_ordre0_INC9[1][1])), calcul_projection(np.mean(data_ordre0_INC9, 0)), c='orange')
    for i in range(8):
        plt.plot(range(len(data_ordre0_LED5[1][1])), calcul_projection(data_ordre0_LED5[i]), '--', c='blue', alpha=0.3)
        plt.plot(range(len(data_ordre0_LED9[1][1])), calcul_projection(data_ordre0_LED9[i]), '--', c='red', alpha=0.3)
        plt.plot(range(len(data_ordre0_INC5[1][1])), calcul_projection(data_ordre0_INC5[i]), '--', c='green', alpha=0.3)
        plt.plot(range(len(data_ordre0_INC9[1][1])), calcul_projection(data_ordre0_INC9[i]), '--', c='orange',
                 alpha=0.3)
    plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])
    plt.title("Projection sur l'axe horizontal")

    plt.subplot(212)
    plt.plot(range(len(data_ordre0_LED5[1])), calcul_projection(np.mean([el.T for el in data_ordre0_LED5], 0)),
             c='blue')
    plt.plot(range(len(data_ordre0_LED9[1])), calcul_projection(np.mean([el.T for el in data_ordre0_LED9], 0)), c='red')
    plt.plot(range(len(data_ordre0_INC5[1])), calcul_projection(np.mean([el.T for el in data_ordre0_INC5], 0)),
             c='green')
    plt.plot(range(len(data_ordre0_INC9[1])), calcul_projection(np.mean([el.T for el in data_ordre0_INC9], 0)),
             c='orange')
    for i in range(8):
        plt.plot(range(len(data_ordre0_LED5[1])), calcul_projection(data_ordre0_LED5[i].T), '--', c='blue', alpha=0.3)
        plt.plot(range(len(data_ordre0_LED9[1])), calcul_projection(data_ordre0_LED9[i].T), '--', c='red', alpha=0.3)
        plt.plot(range(len(data_ordre0_INC5[1])), calcul_projection(data_ordre0_INC5[i].T), '--', c='green', alpha=0.3)
        plt.plot(range(len(data_ordre0_INC9[1])), calcul_projection(data_ordre0_INC9[i].T), '--', c='orange', alpha=0.3)
    plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])
    plt.title("Projection sur l'axe vertical")


def plot_feature(features, data_x=None):
    if isinstance(features, list):
        for func in features:
            fig = plt.figure()
            plt.scatter(range(len(data_ordre0_LED5)), func(data_ordre0_LED5), c='blue')
            plt.scatter(range(len(data_ordre0_LED9)), func(data_ordre0_LED9), c='red')
            plt.scatter(range(len(data_ordre0_INC5)), func(data_ordre0_INC5), c='green')
            plt.scatter(range(len(data_ordre0_INC9)), func(data_ordre0_INC9), c='orange')
            if data_x is None:
                plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])
            else:
                plt.scatter(8, func(data_x), marker='x', color='k')
                plt.legend(['LED5', 'LED9', 'INC5', 'INC9', 'data_x'])
            plt.title(func.__name__)
            plt.show()
    else:
        func = features
        fig = plt.figure()
        plt.scatter(range(len(data_ordre0_LED5)), func(data_ordre0_LED5), c='blue')
        plt.scatter(range(len(data_ordre0_LED9)), func(data_ordre0_LED9), c='red')
        plt.scatter(range(len(data_ordre0_INC5)), func(data_ordre0_INC5), c='green')
        plt.scatter(range(len(data_ordre0_INC9)), func(data_ordre0_INC9), c='orange')
        if data_x is None:
            plt.legend(['LED5', 'LED9', 'INC5', 'INC9'])
        else:
            plt.scatter(8, func(data_x), marker='x', color='k')
            plt.legend(['LED5', 'LED9', 'INC5', 'INC9', 'data_x'])
            plt.title(func.__name__)
        plt.show()



def plot_feature_space(X, y, features, dims=[0, 1, 2]):
    names = np.unique(y)
    colors = ['b', 'r', 'g', 'orange', 'k']
    markers = ['o', 'o', 'o', 'o', 'x']

    if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[-1] == x.shape[-1]:
        fig = plt.figure()

        if X.shape[1] == 2:
            for i, name in enumerate(names):
                plt.scatter(X[y == name, 0], X[y == name, 1], marker=markers[i],color=colors[i])
        else:
            ax = fig.add_subplot(projection='3d')
            ax.set_zlabel(features[dims[2]].__name__)
            for i, name in enumerate(names):
                ax.scatter(X[y == name, dims[0]], X[y == name, dims[1]], X[y == name, dims[2]], marker=markers[i], color=colors[i])

        plt.legend(names)
        plt.xlabel(features[dims[0]].__name__)
        plt.ylabel(features[dims[1]].__name__)
        plt.show()
    else:
        print('burp')
        return 'burp'


def knn(X, x, y, k=2):
    dists = np.sqrt(((X - x) ** 2).sum(axis=1))
    ind = np.argsort(dists)

    unique, counts = np.unique(y[ind[0]], return_counts=True)
    result = unique[np.argmax(counts)]

    plot_feature_space(np.vstack((X, x)), np.hstack((y, "data_X = " + result)), features)

    print(result)
    return result


data = getdata()
data_x = data[0]
features = [calcul_moyenne, calcul_var, calcul_conv]

plot_images()
plot_proj()
#plot_feature(features, data_x)

X = np.array([func(data) for func in features]).T
x = np.array([func(data_x) for func in features]).T
y = np.array(["LED5"] * 8 + ["LED9"] * 8 + ["INC5"] * 8 + ["INC9"] * 8)

X_norm = 1 / X.max(axis=0)
X = X * X_norm
x = x * X_norm

#plot_feature_space(np.vstack((X, x)), np.hstack((y, ["data_x"])), features)

result = knn(X, x, y, k=5)
