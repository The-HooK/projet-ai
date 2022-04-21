import numpy as np
import matplotlib.pyplot as plt

# from skimage import feature
from sklearn.neighbors import KNeighborsClassifier

from mahotas import euler
from skimage.io import imread, imread_collection, concatenate_images
from skimage.color import rgb2gray


def getdata():
    # global data_LED5
    # global data_LED9
    # global data_INC5
    # global data_INC9
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


def calcul_moyenne(images):
    if not isinstance(images, list):
        images = [images]
    return [np.mean(img) for img in images]


def calcul_var(images):
    if isinstance(images, list):
        return [np.var(calcul_projection(img)) for img in images]
    else:
        return np.var(calcul_projection(images))


def calcul_cov(images):
    if isinstance(images, list):
        return [np.cov(calcul_projection(img), calcul_projection(img.T))[1, 0] for img in images]
    else:
        return np.cov(calcul_projection(images), calcul_projection(images.T))[1, 0]


def calcul_conv(images):
    if isinstance(images, list):
        return [sum(calcul_projection(img) * calcul_projection(img.T)) for img in images]
    else:
        return sum(calcul_projection(images) * calcul_projection(images.T))


def calcul_projection(images):
    if isinstance(images, list):
        return [sum(img) for img in images]
    else:
        return sum(images)


def calcul_euler(images):
    if isinstance(images, list):
        return [euler(img >= 0.5) for img in images]
    else:
        return euler(images >= 0.5)


def calcul_barycentre(images):
    proj = calcul_projection(images)
    if isinstance(images, list):
        return [sum(range(len(p)) * p) / len(p) / np.sum(p) for p in proj]
    else:
        return sum(range(len(proj)) * proj) / len(proj)


def calcul_barycentre_T(images):
    if isinstance(images, list):
        proj = calcul_projection([img.T for img in images])
        return [sum(range(len(p)) * p) / len(p) / np.sum(p) for p in proj]
    else:
        proj = calcul_projection(images)
        return sum(range(len(proj)) * proj) / len(proj)


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


def plot_feature_space(X, y, features_names, dims=[0, 1, 2]):
    """
The function plot_feature_space() takes in four arguments:
X (the data), y (the labels for the data), k (the number of neighbors to consider).
It displays the data in a 3D (or 2D) feature space.
    """
    names = np.unique(y)
    colors = ['b', 'r', 'g', 'orange', 'k']
    markers = ['o', 'o', 'o', 'o', 'x']
    global fig

    if isinstance(X, np.ndarray) and X.ndim == 2:  # and X.shape[-1] == x.shape[-1]:
        fig = plt.figure()

        if X.shape[1] == 2:
            for i, name in enumerate(names):
                plt.scatter(X[y == name, 0], X[y == name, 1], marker=markers[i], color=colors[i])
        else:
            ax = fig.add_subplot(projection='3d')
            ax.set_zlabel(features_names[dims[2]])
            for i, name in enumerate(names):
                ax.scatter(X[y == name, dims[0]], X[y == name, dims[1]], X[y == name, dims[2]], marker=markers[i],
                           color=colors[i])

        plt.legend(names)
        plt.xlabel(features_names[dims[0]])
        plt.ylabel(features_names[dims[1]])
        plt.show()
    else:
        print('burp')
        return 'burp'

    return fig


def plot_features_1D(X, y, features_names):
    if not isinstance(features_names, list):
        features_names = [features_names]
    if len(features_names) != X.shape[1]:
        print("youpsiii ! error in plot_features_1D")
        return 'burp'
    plt.figure()
    for i, name in enumerate(features_names):
        plt.subplot(X.shape[1]*100 + 11 + i)
        plt.hlines(1, 0, 1)  # Draw a horizontal line
        plt.eventplot(X[0:7, i], orientation='horizontal', colors='b')
        plt.eventplot(X[8:15, i], orientation='horizontal', colors='r')
        plt.eventplot(X[16:23, i], orientation='horizontal', colors='g')
        plt.eventplot(X[24:31, i], orientation='horizontal', colors='orange')
        plt.title(name)
        plt.axis('off')
    plt.show()


def knn(X, query_points, y, k=2):
    """
This is a k-nearest neighbor classifier. It takes in four arguments:
X (the data), x (the point to classify), y (the labels for the data), and k (the number of neighbors to consider).
It returns the classification for x.
    """
    results = []
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, X.shape[1])
    for i in range(query_points.shape[0]):
        x = query_points[i]
        dists = np.sqrt(((X - x) ** 2).sum(axis=1))
        ind = np.argsort(dists)
        unique, counts = np.unique(y[ind[1:(k+1)]], return_counts=True)
        results.append(unique[np.argmax(counts)])
    return results


def knn_bis(X, x, y, k=2):
    """
This is a k-nearest neighbor classifier. It takes in four arguments:
X (the data), x (the point to classify), y (the labels for the data), and k (the number of neighbors to consider).
It returns the classification for x.
    """
    dists = np.sqrt(((X - x) ** 2).sum(axis=1))
    ind = np.argsort(dists)

    unique, counts = np.unique(y[ind[0]], return_counts=True)

    return y[ind]


data = getdata()
# data_x = data[0]
features = [calcul_moyenne, calcul_var, calcul_conv, calcul_barycentre, calcul_barycentre_T, calcul_euler]
features_names = [func.__name__ for func in features]
dimensions = [3, 4, 5]

# CALCUL ET MISE EN FORME DES DONNEES
X = np.array([func(data) for func in features]).T

# AJOUT DE DONNEES BRUTES (euler)
X_brut_LED = [1, 1, 1, 1, 0, 2, 1, 1, 0, 1, 0, 0, 0, 0, 2, -1]
X_brut_INC = [1, 1, 1, 2, 0, 3, 0, 0, 1, 2, 1, 2, 3, 3, 4, 9]
raw_data = np.concatenate([X_brut_LED, X_brut_INC])
raw_data = raw_data[:, np.newaxis]
features_names.append('euler')
X = np.append(X, raw_data, axis=1)

# AJOUT DE DONNEES BRUTES (TF)
X_brut_LED = [19.0, 19.0, 21.0, 17.0, 17.0, 19.0, 17.0, 15.0, 17.0, 21.0, 19.0, 17.0, 17.0, 15.0, 19.0, 19.0]
X_brut_INC = [9.0, 15.0, 9.0, 11.0, 15.0, 7.0, 11.0, 11.0, 11.0, 15.0, 9.0, 11.0, 11.0, 8.0, 13.0, 9.0]
raw_data = np.concatenate([X_brut_LED, X_brut_INC])
raw_data = raw_data[:, np.newaxis]
features_names.append('largeur spectre')
X = np.append(X, raw_data, axis=1)

# AJOUT DE DONNEES BRUTES (TF2)
X_brut_LED = [246.0, 255.0, 252.0, 249.0, 241.0, 252.0, 258.0, 259.0, 248.5, 233.0, 246.0, 249.0, 241.0, 264.0, 259.0,
              267.0]
X_brut_INC = [190.0, 192.0, 179.0, 182.0, 175.0, 182.0, 182.0, 177.0, 187.0, 178.0, 179.0, 175.0, 185.5, 181.0, 184.0,
              178.0]
raw_data = np.concatenate([X_brut_LED, X_brut_INC])
raw_data = raw_data[:, np.newaxis]
features_names.append('largeur spectre bis')
X = np.append(X, raw_data, axis=1)

# CREATION DES LABELS
y = np.array(["LED5"] * 8 + ["LED9"] * 8 + ["INC5"] * 8 + ["INC9"] * 8)
# y = y np.concatenate([y, ""])

# NORMALISATION DES DONNEES
X_norm = 1 / X.max(axis=0)
X = X * X_norm

# PRESENTATION DES DONNEES
plot_images()
plot_proj()
# plot_feature_space(np.append(X, raw_data, axis=1), y, features_names, dimensions)

# CLASSIFICATION PAR K-NN
# x = np.array([func(data_x) for func in features]).T
# x = X * X_norm
x = X[0]
n_neighbors = 5
result = knn(X, x, y, n_neighbors)
results = knn(X, X, y, n_neighbors)
#results = knn_bis(X, x, y, n_neighbors)

clf = KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)
Z = clf.predict(X)

# AFFICHAGE DES RESULTATS
plot_features_1D(X, y, features_names)
plot_feature_space(np.vstack((X, x)), np.hstack((y, "data_X")), features_names, dimensions)
plt.suptitle("k-NN classification (k=" + str(n_neighbors) + ")", fontsize=16)
plt.title("data_x = " + result[0], fontsize=12)
