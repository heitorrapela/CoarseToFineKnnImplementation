# Nearest Feature Line - Machine Learning Project - CIn/UFPE
# Members:
# Gabriel de Franca Medeiros
# Gabriel Marques Bandeira
# Heitor Rapela Medeiros 
import numpy as np
import pandas as pd
import plotly as py
from plotly.graph_objs import *
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
import glob
from PIL import Image
import itertools


# Separate data from palmprint dataset
def get_x_std(X):
    return StandardScaler().fit_transform(X)


# mi : Element of Real Numbers
def calculate_mi(x1, x2, x):
    aux1 = np.dot(x - x1, x2 - x1)
    aux2 = np.dot(x2 - x1, x2 - x1)
    if (aux2 != 0):
        mi = aux1 / float(aux2)
    else:
        mi = 0.0
    return mi


# p : projection point of x in x1x2
def calculate_p(x1, x2, mi):
    p = x1 + mi * (x2 - x1)
    return p


def calculate_distance_x1x2(x, p):
    return LA.norm(x - p)


def distance_nearestFeatureLine(x1, x2, x):
    mi = calculate_mi(x1, x2, x)
    p = calculate_p(x1, x2, mi)
    dist = calculate_distance_x1x2(x, p)
    return dist


################## Data pre-processing ###########################
# Multispectral Palm Print DataBase
# Class quantity
numCat = 320
# Number of attributes = 400
att = 400

# Open data file
# path = '/home/bandeira/Documents/ufpe/am/projeto/CoarseToFineKnnImplementation/multispectral palmprint/'
path = '/home/bandeira/Documents/ufpe/am/projeto/CASIA-PM-V1-resized/'
files = []
labels = []
datas = []
for fn in glob.glob(path + "*.jpg"):
    files.append(fn)
    labels.append(int(fn[fn.rfind('/') + 1:].split('_')[0]) - 1)
    img = Image.open(fn)
    datas.append(list(img.getdata(band=0)))

# Separate Train Data #
X_semStd = np.array(datas[:int(len(datas) * 0.75)])
class_label_tranning = np.array(labels[:int(len(labels) * 0.75)])
X = get_x_std(X_semStd)

# Separate Test Data #
Y_semStd = np.array(datas[int(len(datas) * 0.75):])
class_label_test = np.array(labels[int(len(labels) * 0.75):])
Y = get_x_std(Y_semStd)

sortedByClassTrainingData = []
for n_class in range(0, numCat):
    sortedByClassTrainingData.append(X[class_label_tranning == n_class])

###################################################################

K = [1, 5, 7]
K_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

counter = 0
# Get each instance of the test from db
for x in Y:
    print str(counter) + '/' + str(len(Y))
    # Y shape : (att,) change to (1,att)
    x = np.array([x])
    dist = []
    for c in range(0, numCat):
        aux = sortedByClassTrainingData[c]
        if len(aux) > 0:
            class_label_tranning = c
            Aux = get_x_std(aux)
            teste = list(itertools.product(Aux, Aux))

            for v in range(0, len(teste)):
                x1 = np.asarray(teste[v][0])
                x2 = np.asarray(teste[v][1])
                dist_aux = distance_nearestFeatureLine(x1, x2, x)
                dist.append([c, dist_aux])
    dist_ = sorted(dist, key=lambda x: x[1])

    for k in range(len(K)):
        # get the k first elements in the list
        aux_ar = dist_[:K[k]]
        # get the classes
        classes = [x[0] for x in aux_ar]
        classes = np.array(classes)
        # get the most common class in the array
        counts = np.bincount(classes)
        result = np.argmax(counts)
        # compare with the known class
        if result == class_label_test[counter]:
            K_rate[k] = K_rate[k] + 1
    counter = counter + 1
# calculating the rate
print [x / float(len(Y)) for x in K_rate]

# print 'Best option: ' + str(best_option[0]) + '%   -   N = ' + str(best_option[1]) + '; K = ' + str(best_option[2])
# x = N
# y = taxa
# title = 'CFKNNC - K: ' + str(K[0]) + ''
# py.offline.plot({
#     "data": [Scatter(x=x, y=y)],
#     "layout": Layout(title=title,
#                      yaxis=dict(range=[0, 100],
#                                 title='%'),
#                      xaxis=dict(title='N')
#                      )
# })
