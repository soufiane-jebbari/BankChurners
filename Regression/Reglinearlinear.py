# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 18:04:40 2021

@author: nouhaila
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model

#=========================== Part1 : Datasets ==========================

data = pd.read_csv("student-mat.csv", sep=";")
# Since our data is seperated by semicolons we need to do sep=";"

#print(data.head()) 
#print just the first 5 rows

data1 = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#select attributes we want to use

X = np.array(data1.drop('G3', axis=1))
#all the columns of data  are being stored in the X variable except the "G3" column
#Features
y = np.array(data1['G3']).reshape(395,1)
#Labels


G1 = np.array(data1['G1']).reshape(395,1)
G2 = np.array(data1['G2']).reshape(395,1)
studytime = np.array(data1['studytime']).reshape(395,1)
failures = np.array(data1['failures']).reshape(395,1)
absences = np.array(data1['absences']).reshape(395,1)


X1 = np.hstack((G1, G2, studytime, failures, absences, np.ones(y.shape)))
theta = np.random.randn(6, 1)

#========================= Part2 : Funtions ==============================

#le modèle
def modele(x, theta):
   return x.dot(theta)

#l'erreur
def err_function(x, y, theta):
   z = len(y)
   return 1/(2*z) * np.sum((modele(x, theta) - y)**2)

#gradient
def grad(x, y, theta):
    z = len(y)
    return 1/z * x.T.dot(modele(x, theta) - y)

#desente du gradient
def gradient_descent(x, y, theta, alpha, n_iterations):
    err_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
          err_history[i] = err_function(x, y, theta)
          theta = theta - alpha * grad(x, y, theta)
    return theta, err_history


#========================== Part3 = plot data ============================


n_iterations = 1000
alpha = 0.001


theta_optimal, err_history = gradient_descent(X1, y, theta, alpha, n_iterations)
#plt.plot(theta_optimal)

## La prédiction
predictions = modele(X1, theta_optimal)

## Affiche des résultats
L=np.hstack((G1, G2, studytime, failures, absences))    
K = ['G1', 'G2', 'studytime', 'failures', 'absences']
for i in range(5):    
      plt.scatter(L[:,i], y)
      plt.scatter(L[:,i], predictions, c='r')
      plt.title("plot number " + str(i))
      plt.xlabel(K[i])
      plt.ylabel('G3')
      plt.show()

##afficher l'historique de l'erreur
plt.plot(range(n_iterations), err_history)
plt.show()


#========================= Accuracy =============================
##Best score
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X1, y, test_size=0.1)
    #We will use 90% of our data to train and the other 10% to test. 
    #The reason we do this is so that we do not test our model on data that it has already seen.
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))
    
    
#============================ But du projet =========================

# linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)
# print(acc)
# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)

# predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])


#======================= Affichage en 2D ===============================
##2d
fig = plt.figure()
ax=plt.axes(projection='3d')
ax.scatter(G1, G2, y, c='r')
ax.scatter(G1, G2, predictions, c='g')
    



