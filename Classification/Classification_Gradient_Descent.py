# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 20:38:17 2021

@author: admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#======================etape1:preparing our variables=======================================================

data_pd = pd.read_csv('BankChurners.csv')#extraire les donnees du fichier csv

del data_pd["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"]
del data_pd["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]
#deleting the last two columns as in te discreption
del data_pd["CLIENTNUM"]#the id has nothing to do with  the computing

#let's encode the dummy variables
#nominal encoding:

#the gender variable:
    
dummy_gender=pd.get_dummies(data_pd.Gender)
#we concatenate the new encoded columns with our dataframe in a new variable data_up1
data_up1=pd.concat([data_pd,dummy_gender],axis='columns')
#we delete the nominal column as it is encoded
del data_up1["Gender"]
#we delete one dummy not to fall in dummies trap
del data_up1["F"]

#the marital status variable:
    
dummy_marital_status=pd.get_dummies(data_up1.Marital_Status)
data_up2=pd.concat([data_up1,dummy_marital_status],axis='columns')
del data_up2["Marital_Status"]
del data_up2["Unknown"]

#the card category variable:
    
dummy_card_category=pd.get_dummies(data_pd.Card_Category)
data_up3=pd.concat([data_up2,dummy_card_category],axis='columns')
del data_up3["Card_Category"]
del data_up3["Platinum"]

#ordianl variables

# w we create a function that gives the digfferent values of the column
def divide_column(column):   
    L=[]
    for i in column:
        if i not in L:
            L.append(i)
    return L

values_education_level= divide_column(data_up3["Education_Level"])
values_income_category= divide_column(data_up3["Income_Category"])

#encoding education_level column:
    
nuemric_education_level={ "Unknown":0,
                         "Uneducated":1,
                         "High School":2,
                         "College":3,
                         "Graduate":4,
                         "Post-Graduate":5,
                         "Doctorate":6}

#we make a copy of the education level to work on it as it is not possible to work on the original column
education_level_copy=pd.DataFrame({"education_level_numeric":np.zeros(len(data_up3["Education_Level"]))})

#we convert each nominal value with its numerical value
for i in range(len(data_up3["Education_Level"])):
    for j in nuemric_education_level.keys():
        if data_up3["Education_Level"][i]==j:
            education_level_copy["education_level_numeric"][i]=nuemric_education_level[j]

#we concatenate the new column and delete the original one
data_up4=pd.concat([data_up3,education_level_copy],axis='columns')
del data_up4["Education_Level"]

#encoding income category:
    
numeric_income={'$60K - $80K':70000,
                'Less than $40K':40000,
                '$80K - $120K':100000,
                '$40K - $60K':50000,
                '$120K +':120000,
                'Unknown':0}
income_copy=pd.DataFrame({"income_numeric":np.zeros(len(data_up4["Income_Category"]))})
for i in range(len(data_up4["Income_Category"])):
    for j in numeric_income.keys():
        if data_up3["Income_Category"][i]==j:
            income_copy["income_numeric"][i]=numeric_income[j]
            
data_up5=pd.concat([data_up4,income_copy],axis='columns')
del data_up5["Income_Category"]

#converting y to numeric values
y_pd = data_up5["Attrition_Flag"]
numeric_result={'Attrited Customer':0,'Existing Customer':1}
y_num=np.zeros((len(y_pd),1))
for i in range(len(y_pd)):
    for j in numeric_result.keys():
        if y_pd[i]==j:
            y_num[i]=numeric_result[j]
            
X_pd = data_up5.drop(["Attrition_Flag"],axis=1)
            
#now we have our numerical y and numerical X we can begin our computing

#===================etape2: the computing========================================================================

#creating our parametrs X and y as arrays 
X_brute = np.array(X_pd)
y = np.array(y_num)
Dc_X = X_brute.shape[0]# we store the number of rows in Dc_X to use it in creating our X
X = np.concatenate((np.ones((Dc_X,1)),X_brute),axis=1)
Dl_X=X.shape[1] #we store number of columns in Dl_X to use it in creating theta
theta = np.random.randn(Dl_X, 1)
#here
#feature scalling:

def my_mean(x):
    s=0
    s=sum(x)/len(x)
    return s

def feature_normalis(x):
    x_new=np.zeros((x.shape[0],x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            x_new[j,i] = ( x[j,i] - my_mean(x[:,i]) )/(max(x[:,i]) - min(x[:,i]))
    return x_new  

X_fn = feature_normalis(X[:,1:])
X_f = X = np.concatenate((np.ones((Dc_X,1)),X_fn),axis=1)#finally we get our X_f as a final X


#the plotting attempt===================================================================================
pos=(y_num==1).ravel()
neg=(y_num==0).ravel()

#we create a function to plot the pairs of the features to find the best fetaures to describe our dataset
def plot_features(x1,x2,pos,neg,i,j):
    fig, ax = plt.subplots()
    plt.scatter(x1[pos], x2[pos], c='r', marker='x', label='Existing Customer')
    plt.scatter(x1[neg], x2[neg], c='b', marker='.', label='Attrited Customer')
    ax.set_title("test " + str(i) + ' and ' + str(j))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2 ')
    plt.show()

for i in range(X_f.shape[1]):
    for j in range(X_f.shape[1]):
        if i!=j:
            
            plot_features(X_f[:,i],X_f[:,j],pos,neg,i,j)
            
#choosing the best pair of features to visualize

mini_pos=pos[9712:]
mini_neg=neg[9712:]
fig, ax = plt.subplots()
plt.scatter(X_f[9712:,11][mini_pos], X_f[9712:,12][mini_pos], c='r', marker='x', label='Existing Customer')
plt.scatter(X_f[9712:,11][mini_neg],X_f[9712:,12][mini_neg] , c='b', marker='.', label='Attrited Customer')
ax.set_title("data plotting")
ax.set_xlabel('Total_Trans_Amt')
ax.set_ylabel('Total_Trans_Ct')
ax.legend()
plt.show()

            

    
############################################################
#model
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#cost function
def erreur(theta, X, y):
    m = len(y)
    first = np.multiply(-y, np.log(sigmoid(X.dot(theta))))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X.dot(theta))))
    return np.sum(first - second) / m

#grad
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(sigmoid(X.dot(theta)) - y)

#desente du gradient 
def gradient_descent(X, y, theta, alpha, n_iterations):
    err_history = np.zeros(n_iterations) 
    for i in range(0, n_iterations):
        theta = theta - alpha * grad(X, y, theta) 
        err_history[i] = erreur(theta,X,y)         
    return theta, err_history
#######################################################
#predictions
n_iterations = 5000
alpha = 0.09

theta_optimal, err_history = gradient_descent(X_f, y, theta, alpha, n_iterations)

#showing the degradation of the cost function in term of the iterations number
plt.plot(range(n_iterations),err_history)
plt.title("la variation de la cost functon en fonction des iterations")
plt.ylabel("cost function")
plt.xlabel("iterations")
plt.show()



def predict(theta, X):
    probability = sigmoid(X.dot(theta))
    return [1 if x >= 0.5 else 0 for x in probability]

predictions = predict(theta_optimal, X_f)

#=============================================================================
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in  zip(predictions, y)]

accuracy = (sum(correct)/len(correct))*100
print ('accuracy = {0}%'.format(accuracy))















            



