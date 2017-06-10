# Coarse to fine KNN - Machine Learning Project - CIn/UFPE
# Members:
# Gabriel de França Medeiros
# Gabriel Marques Bandeira
# Heitor Rapela Medeiros 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Separate data from breastTissue dataset
def separateData(data):
	# y = Class 
	y = data.ix[:,0].values
	# X = Data split in att ; Matrix : qtdElement x att
	X = data.ix[:,1:att+1].values
	X_std = StandardScaler().fit_transform(X)

	return y,X,X_std

################## Data pre-processing ###########################
# Breast Tissue DataBase
# Class quantity
numCat = 6
# Number of attributes = 9
att = 9

# Open data file
data = pd.ExcelFile("breastTissue/BreastTissue.xls")
data.sheet_names

# Open data training set in pandas frame
dataTraining = data.parse("Training")
dataTraining.head()

# Open data test set in pandas frame
dataTest = data.parse("Test")
dataTest.head()

# Separate Train Data #
class_label_tranning, X_semStd, X = separateData(dataTraining)

# Separate Test Data #
class_label_test, Y_semStd, Y = separateData(dataTest)

################## CFKNNC implementation #########################
# N value of CFKNNC : K <= N
N = 10

# K value of CFKNNC : K <= N
K = 5

# mi : constant value in CFKNNC
mi = 0.15

###################################################################
Y = Y[0]
print Y
Y = np.array([Y])
print Y
print Y.shape

print X.shape
XT = X.T
XTX = np.dot(XT,X)
print XTX.shape
I = np.identity(XTX.shape[0])
miI = mi*I
print (XTX + miI).shape
invs = np.linalg.inv(XTX + miI)
print "inv: ", invs.shape
invsXT = np.dot(invs,XT)
print invsXT.shape
print "XT: ", XT.shape
print Y.shape

#gamma = np.dot(invsXT,Y)
#print gamma
#print X.shape

'''
print X[0:2]
print X.shape
print X.T[0:2]
print X.T.shape
print np.dot(X,X.T)[0:2]
print np.dot(X,X.T).shape
'''