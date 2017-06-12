# Coarse to fine KNN - Machine Learning Project - CIn/UFPE
# Members:
# Gabriel de Franca Medeiros
# Gabriel Marques Bandeira
# Heitor Rapela Medeiros 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit
#import plotly.plotly as py
#from plotly.graph_objs import *
#import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Separate data from breastTissue dataset
def separateData(data):
	# y = Class 
	y = data.ix[:,0].values
	# X = Data split in att ; Matrix : qtdElement x att
	X = data.ix[:,1:att+1].values
	X_std = StandardScaler().fit_transform(X)

	return y,X,X_std

def calculate_gamma(X, Y):
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

	return gamma

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

# iterate test set
y_class = []
for y in Y:
	# Get one instance of the test from db
	# ps: we need to change for a loop (for) in numbers of test instance
	# Y = Y[0]
	# print y.shape
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
	# print error_y_ord.shape
	# print error_y_ord[0,:]
	Z = X[:, error_y_ord[0:N, 0].astype(int)]
	class_z = class_label_tranning[error_y_ord[0:N, 0].astype(int)]

	new_gamma = calculate_gamma(Z, y)
	error_y = []
	for i in range(len(new_gamma)):
		error_y.append(np.linalg.norm(y[:,0]-new_gamma[i][0]*Z[:,i].T))
	indexes = np.array(range(len(error_y)))
	t = np.c_[indexes, error_y]
	error_y_ord = np.array(sorted(t, key=lambda a_entry: a_entry[1]))
	print error_y_ord
	print class_z
	classes = class_z[error_y_ord[0:K, 0].astype(int)]

	print classes
	data = Counter(classes)
	print data.most_common(4)
	y_class.append(data.most_common(1)[0][0])

ok = 0
for i in range(len(class_label_test)):
	print "Y: " + str(class_label_test[i]) + "; identificado como " + str(y_class[i])
	# print class_label_test[i]
	# print y_class[i]
	if class_label_test[i] == y_class[i]:
		ok = ok + 1

print "Taxa: " + str(float(ok)/len(class_label_test)*100) + "%"

# # Passos:
# 1) ordenar o gamma e salvando a posicao, pq ele eh a nova distancia (como se fosse a euclideana)
# 2) pegar os n menores, e calcular a funcao de erro la (eu nao lembro pra que a gente vai usar a funcao de erro)
# 3) repetir os mesmos calculos que foram feitos para o gamma, para o segundo filtro (acho que eh Z agora)
# 4) fazer o mesmo passo que 1) e 2), pegando os k menores
# 5) pronto, temos nossa representacao do caso de teste, pelos k prototipos mais proximos, podemos agora usar o knn e comparar a label correta                        
# 6) testar com dois bancos do UCI, junto com os KNN e suas implementacoes
# 7) testar com os datasets propostos pelo cara
# 
# #Dificuldade:
# 1) eh o mais chato, e que perde mais tempo
# 2) facil, so da um sort e selecionar os n menores
# 3) ja foi feito no codigo, so criar uma funcao e colocar o codigo ja implementado, pq esse novo filtro eh igual ao primeiro, so que so muda a entrada
# 4) repetir os passos 1 e 2
# 5) facil, so colocar num "for", pra pegar as varias entradas, pq so ta pegando a primeira
# 6) Complicado, vamos ter que ver como implementar os outros KNN
# 7) Complicado, arrumar e tratar a entrada pro codigo da gente
