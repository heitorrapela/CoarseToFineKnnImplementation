# Coarse to fine KNN - Machine Learning Project - CIn/UFPE
# Members:
# Gabriel de Franca Medeiros
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
# The notation used in the article is transpose of our database
# so att : its lines and the instace : its columns, so we need to transpose
# our data

# Get one instance of the test from db
# ps: we need to change for a loop (for) in numbers of test instance
Y = Y[0]
print Y.shape
# Y shape : (att,) change to (1,att) 
Y = np.array([Y])
# Convert Y shape to (att,1), article representation of data
Y = Y.T

# Convert our database representation to article representation (Transpose input data) 
# Matrix our representation: (instance,att)
# Article representation: (att, instance) 
X = X.T

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
# Calculate gamma = power((XT*X + mi*I),-1)*XT*Y
gamma = np.dot(invsXT,Y)