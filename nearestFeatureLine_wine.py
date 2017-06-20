# Nearest Feature Line - Machine Learning Project - CIn/UFPE
# Members:
# Gabriel de Franca Medeiros
# Gabriel Marques Bandeira
# Heitor Rapela Medeiros 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import time as time
#import plotly.plotly as py
#from plotly.graph_objs import *
#import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from numpy import linalg as LA

# Separate data from breastTissue dataset
def separateData(data):
	# y = Class 
	y = data.ix[:,0].values
	# X = Data split in att ; Matrix : qtdElement x att
	X = data.ix[:,1:att+1].values
	X_std = StandardScaler().fit_transform(X)

	return y,X,X_std

# mi : Element of Real Numbers
def calculate_mi(x1,x2,x):
	aux1 = np.dot(x-x1,x2-x1)
	aux2 = np.dot(x2-x1,x2-x1) 
	if(aux2 != 0):
		mi = aux1/float(aux2)
	else:
		mi = 0.0
	return mi

# p : projection point of x in x1x2
def calculate_p(x1,x2,mi):
	p = x1 + mi*(x2-x1)
	return p

def calculate_distance_x1x2(x,p):
	return LA.norm(x-p)

def distance_nearestFeatureLine(x1,x2,x):
	mi = calculate_mi(x1,x2,x)
	p = calculate_p(x1,x2,mi)
	dist = calculate_distance_x1x2(x,p)
	return dist

################## Data pre-processing ###########################
# Number of att = 13
att = 13 
# Class quantity
numCat = 3  

# Data Base: Wine recognition data	- Class + 13 att	
dataTraining = pd.read_csv('wine/wine_training.txt', sep=",", header = None)
# To help manipulate, my hashtable will be changed to 0 .. 12
dataTraining.columns = ["Class",0,1,2,3,4,5,6,7,8,9,10,11,12]

# Data Base: Wine recognition data	- Class + 13 att	
dataTest = pd.read_csv('wine/wine_test.txt', sep=",", header = None)
# To help manipulate, my hashtable will be changed to 0 .. 12
dataTest.columns = ["Class",0,1,2,3,4,5,6,7,8,9,10,11,12]

sortedByClassTrainingData = []
for n_class in range(0,numCat):
	sortedByClassTrainingData.append(dataTraining[dataTraining["Class"] == n_class])

K = range(1,10)
K_rate = [0,0,0,0,0,0,0,0,0,0]

# Separate Test Data #
class_label_test, Y_semStd, Y = separateData(dataTest)
counter = 0
# Get each instance of the test from db
for x in Y:
	# Y shape : (att,) change to (1,att) 
	x = np.array([x])
	dist = []
	for c in range(0,numCat):
		aux = sortedByClassTrainingData[c]
		class_label_tranning, aux_semStd, Aux = separateData(aux)
		for i in range(len(Aux)):
			for j in range(len(Aux)):
				if((i != j) and (i < j)):
					x1 = Aux[i]
					x2 = Aux[j]	
					dist_aux = distance_nearestFeatureLine(x1,x2,x)
					dist.append([class_label_tranning[0],dist_aux])
	dist_ = sorted(dist, key=lambda x: x[1])
	
	for k in range(len(K)):
		#get the k first elements in the list
		aux_ar = dist_[:k+1]
		#get the classes
		classes = [x[0] for x in aux_ar]
		classes = np.array(classes)
		#get the most common class in the array
		counts = np.bincount(classes)
		result = np.argmax(counts)
		#compare with the known class
		if result == class_label_test[counter]:
			K[k] = K[k] + 1
	counter = counter + 1
#calculating the rate 
print [x/float(len(Y)) for x in K]