import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

cars = pd.read_csv("Data files/car+evaluation/car.data")

x = cars[['buying','maint','safety']].values
y = cars['class']
y1 = cars[['class']]
#print(y,y1)
le = LabelEncoder()
#x
for i in range(len(x[0])):
    x[:, i] = le.fit_transform(x[:, i])
new_mappiny = {'unacc':0,'acc':1,'good':2,'vgood':3}
#y
y = y.map(new_mappiny)
#print(y)

#print(y)
#print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4)
#print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
knn = neighbors.KNeighborsClassifier(n_neighbors=25)
knn.fit(x_train,y_train)
predicted_value = knn.predict(x_test)
print(f"Accuracy {metrics.accuracy_score(y_test,predicted_value)}")
print(f"predicted value : {predicted_value}")
print(f"Actual value : {y_test} ")
