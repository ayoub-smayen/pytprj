# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:39:03 2020

@author: LENOVO

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# make predictions
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
"""

import numpy as np

import scipy as sc
import sympy as sp
import matplotlib.pyplot as plt 
import pandas  as pd

import  sklearn as sk
import csv
l1 = [1,2,3]
l2 = l4 = [55,23,98]
l3 = l2.copy()

h = np.array([l1,l2,l3])


print(h)


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
with open("ml_datas.txt","w") as f:
    f1 = csv.writer(f,delimiter=',')
    f1.writerow(dataset["class"])

    
    
print(dataset.shape)
describe =dataset.describe()
print(describe)
grp =dataset.groupby('class').size()
print(grp)
print(dataset["class"])
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
dataset.hist()
plt.show()
