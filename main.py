import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mahotas import euler
from skimage.io import imread
from skimage.color import rgb2gray



#OUTILS ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
    data_ordre0_INC5 = [np.pad(data[:66,:66], pad_width=27, constant_values=0) for data in data_INC5]
    data_ordre0_INC9 = [np.pad(data[:66,:66], pad_width=27, constant_values=0) for data in data_INC9]
    data_ordre0_INC5 = [data[:66, :66] for data in data_INC5]
    data_ordre0_INC9 = [data[:66, :66] for data in data_INC9]

    return data_ordre0_LED5 + data_ordre0_LED9 + data_ordre0_INC5 + data_ordre0_INC9


def crop(image):
    if isinstance(image, list):
        return [img[510:630, 770:890] for img in image]
    else:
        return image[510:630, 770:890]


def calcul_projection(images):
    if isinstance(images, list):
        return [sum(img) for img in images]
    else:
        return sum(images)


#FEATURES /////////////////////////////////////////////////////////////////////////////////////////////////////////////
def calcul_moyenne(images):
    if not isinstance(images, list):
        images = [images]
    return [np.mean(img) for img in images]


def calcul_conv5(images):
    if not isinstance(images, list):
        images = [images]
    return [np.sum(img*data_ordre0_LED5) for img in images]


def calcul_conv9(images):
    if not isinstance(images, list):
        images = [images]
    return [np.sum(img*data_ordre0_LED9) for img in images]


def calcul_var(images):
    if isinstance(images, list):
        return [np.var(calcul_projection(img)) for img in images]
    else:
        return np.var(calcul_projection(images))


def calcul_var_T(images):
    if isinstance(images, list):
        return [np.var(calcul_projection(img.T)) for img in images]
    else:
        return np.var(calcul_projection(images.T))


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


def calcul_width_TF(liste_images, ratio = 2/3):
    """
    fct permettant de calculer la taille de la tache centrale de la tf d'une image
    (on calcule la TF, puis on regarde seulement la ligne du centre
    et on s'interesse a la premiere fois (en partant du centre) ou la valeur du module chute d'une certaine valeurs

    input : liste d'images. ATTENTION SI IMAGE SEULE IL FAUT LA METTRE DANS UNE LISTE QD MEME !
            ratio : ratio par rapport auquel on calcule l'epaisseur de la courbe: conseil : 2/3 pour img entiere
                                                                                            6/7 pour img crop0
    output : liste des taille des TF des images entrées
    """

    #liste_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in liste_images]
    results = []

    center = int(len(liste_images[0])/2)#indice centre, en gnl nbr entier, ie ligne centrales sont : center et center +1

    for image in liste_images:  #on applique le calcul sur chaque image d'une liste directement dans la fct
        TF = np.log(abs(fft.fftshift(fft.fft2(image)))) #transformee de Fourier de l'image
        TF = TF[center-49:center+51,:] #on s'intéresse à la taille de la tache centrale, on prend des lignes au centre
        TF = [np.mean(TF[:,i]) for i in range(len(TF[0]))]  #moyenne des 100 lignes choisies pour limiter bruit

        #plt.figure()
        #plt.plot(TF)         #(to be used for visual undersanding)
        #plt.show()

        #maintenant, il s'agit de mesurer la taille de la tache on prendre la taille non pas à mi-hauteur mais à 2/3
        max = np.max(TF)
        y = max*ratio #hauteur à laquelle on va mesurer l'épaisseur de la TF

        index_max = TF.index(max)
        TF = [TF[:index_max],TF[index_max:]] #on separe la TF en une phase de montée globale et une phase de descente

        rough_mean_inter = [] #cette liste contiendra deux valeurs :
        # les abscices d'intersection entre la courbe de la TF et la droite y = max*(2/3)
        # pendant les phases de croissance et de décroissance de la TF

        for half_TF in TF:
            tmp = [] #cette liste va stocker toutes les intersections entre la courbe de la (demi) TF et la droite y = max*(2/3)
            for index_x in range(len(half_TF)-1):
                if half_TF[index_x] < y and y < half_TF[index_x + 1] or half_TF[index_x] > y and y > half_TF[index_x + 1]:
                    milieu = 0.5*(index_x+index_x+1)
                    tmp.append(milieu)
            if not tmp:
                tmp.append(len(half_TF)-1)
            rough_mean_inter.append(0.5*(tmp[0]+tmp[-1]))

        rough_mean_inter[1] += len(TF[0]) # en coupant la TF, l'axe x est faussé sur la seconde moitié,
                                          # le centre devient l'indice 0 donc il faut ajouter le décalage


        results.append(rough_mean_inter[1]-rough_mean_inter[0])
    return results


# AFFICHAGE ///////////////////////////////////////////////////////////////////////////////////////////////////////////
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

    if isinstance(X, np.ndarray) and X.ndim == 2:  # and X.shape[-1] == x.shape[-1]:
        fig = plt.figure()

        if X.shape[1] == 2:
            for i, name in enumerate(names):
                dims = [0, 1]
                plt.grid(True, alpha=0.75)
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
        plt.subplot(X.shape[1], 1, i+1)
        plt.hlines(1, 0, 1)  # Draw a horizontal line
        plt.eventplot(X[0:8, i], orientation='horizontal', colors='b')
        plt.eventplot(X[8:16, i], orientation='horizontal', colors='r')
        plt.eventplot(X[16:24, i], orientation='horizontal', colors='g')
        plt.eventplot(X[24:32, i], orientation='horizontal', colors='orange')
        plt.title(name)
        plt.axis('off')
    plt.show()


# MODELE //////////////////////////////////////////////////////////////////////////////////////////////////////////////
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


data = getdata()
# data_x = data[0]

features = [calcul_moyenne, calcul_var, calcul_var_T, calcul_euler, calcul_barycentre,
            calcul_barycentre_T, calcul_width_TF]
features_names = [func.__name__ for func in features]
dimensions = [3, 4, 5]

# CALCUL ET MISE EN FORME DES DONNEES
X = np.array([func(data) for func in features]).T



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


clf = KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)
Z = clf.predict(X)

# AFFICHAGE DES RESULTATS
plot_features_1D(X, y, features_names)
#plot_feature_space(np.vstack((X, x)), np.hstack((y, "data_X")), features_names, dimensions)
#plt.suptitle("k-NN classification (k=" + str(n_neighbors) + ")", fontsize=16)
#plt.title("data_x = " + result[0], fontsize=12)

print(sum(results==y)/0.32)
