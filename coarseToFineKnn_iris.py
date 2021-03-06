# Coarse to fine KNN Classifier - Machine Learning Project - CIn/UFPE
#
# Members:
# Gabriel de Franca Medeiros (gfm@cin.ufpe.br)
# Gabriel Marques Bandeira (gmb@cin.ufpe.br)
# Heitor Rapela Medeiros (hrm@cin.ufpe.br)
#
# Based on Yong Xu algorithm proposed in http://dx.doi.org/10.1016/j.patrec.2013.01.028 (last acessed on 19/june/2017)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit
# import plotly.plotly as py
import plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


# Separate data from Iris dataset
def separateData(data):
	# y = Class 
	y = data.ix[:,0].values
	# X = Data split in att ; Matrix : qtdElement x att
	X = data.ix[:,1:att+1].values
	X_std = StandardScaler().fit_transform(X)

	return y,X,X_std


# Calculate gamma = power((XT*X + mi*I),-1)*XT*Y
def calculate_gamma(X, Y):
	# Calculate power((XT*X + mi*I),-1)*XT
	invsXT = calculate_invs_xt(X)
	# Calculate gamma = power((XT*X + mi*I),-1)*XT*Y
	gamma = np.dot(invsXT,Y)

	return gamma


# Calculate power((XT*X + mi*I),-1)*XT
def calculate_invs_xt(X):
	# Calculate XT 
	XT = X.T
	# Calculate XT*X
	XTX = np.dot(XT,X)
	# Calculate I, same dimension of XT*X lines
	I = np.identity(XTX.shape[0])
	# Calculate mi*I
	miI = mi*I
	# Calculate inverse of (XT*X + mi*I)
	invs = np.linalg.inv(XTX + miI)
	# Calculate power((XT*X + mi*I),-1)*XT
	invsXT = np.dot(invs,XT)

	return invsXT


################## Data pre-processing ###########################
# Number of att = 4
att = 4
# Class quantity
numCat = 3  

# Data Base: Iris Dataset
dataTraining = pd.read_csv('iris/iris_train.txt', sep=",", header = None)
# To help manipulate, hashtable
dataTraining.columns = ["Class",0,1,2,3]

# Data Base: Iris Dataset
dataTest = pd.read_csv('iris/iris_test.txt', sep=",", header = None)
# To help manipulate, hashtable
dataTest.columns = ["Class",0,1,2,3]

# Separate Train Data #
class_label_tranning, X_semStd, X = separateData(dataTraining)

# Separate Test Data #
class_label_test, Y_semStd, Y = separateData(dataTest)

###################################################################
# The notation used in the article is transpose of our database
# so att : its lines and the instace : its columns, so we need to transpose
# our data
# Convert our database representation to article representation (Transpose input data) 
# Matrix our representation: (instance,att)
# Article representation: (att, instance) 
X = X.T

################## CFKNNC implementation #########################
# N value of CFKNNC : K <= N
N = [10,20,30,40,50,60,70]

# K value of CFKNNC : K <= N
K = [1,2,3,4,5]

# mi : constant value in CFKNNC
mi = 0.01

best_option = [0, N[0], K[0]]

# Calculate power((XT*X + mi*I),-1)*XT
invsXT = calculate_invs_xt(X)
taxa = []

for n in N:
	for k in K:
		# Iterate test set
		y_class = []
		# Get each instance of the test from db
		for y in Y:
			# Y shape : (att,) change to (1,att) 
			y = np.array([y])
			# Convert Y shape to (att,1), article representation of data
			y = y.T

			# Calculate gamma = power((XT*X + mi*I),-1)*XT*Y
			gamma = np.dot(invsXT,y)

			error_y = []
			for i in range(len(gamma)):
				error_y.append(np.linalg.norm(y[:,0]-gamma[i][0]*X[:,i].T))
			
			indexes = np.array(range(len(error_y)))
			t = np.c_[indexes, error_y]
			error_y_ord = np.array(sorted(t, key=lambda a_entry: a_entry[1]))
			Z = X[:, error_y_ord[0:n, 0].astype(int)]
			class_z = class_label_tranning[error_y_ord[0:n, 0].astype(int)]

			new_gamma = calculate_gamma(Z, y)
			error_y = []
			for i in range(len(new_gamma)):
				error_y.append(np.linalg.norm(y[:,0]-new_gamma[i][0]*Z[:,i].T))
			indexes = np.array(range(len(error_y)))
			t = np.c_[indexes, error_y]
			error_y_ord = np.array(sorted(t, key=lambda a_entry: a_entry[1]))
			classes = class_z[error_y_ord[0:k, 0].astype(int)]

			data = Counter(classes)
			y_class.append(data.most_common(1)[0][0])

		ok = 0
		for i in range(len(class_label_test)):
			# print "Y label: " + str(class_label_test[i]) + "; identified as " + str(y_class[i])
			if class_label_test[i] == y_class[i]:
				ok = ok + 1

		taxa.append(float(ok)/len(class_label_test)*100)
		if taxa[-1] > best_option[0]:
			best_option[0] = taxa[-1]
			best_option[1] = n
			best_option[2] = k
		print "N = " +str(n) + "; K = " + str(k) + "\tTaxa: " + str(taxa[-1]) + "%"

print 'Best option: ' + str(best_option[0]) + '%   -   N = ' + str(best_option[1]) + '; K = ' + str(best_option[2])
x = N
y = taxa
title = 'CFKNNC - K: ' + str(K[0]) + ''
py.offline.plot({
		"data": [Scatter(x=x, y=y)],
		"layout": Layout(title=title,
						yaxis=dict(range=[0, 100],
									title='%'),
						xaxis=dict(title='N')
						)
	})

# 001_l_460_01.jpg