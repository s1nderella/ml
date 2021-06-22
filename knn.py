import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.preprocessing import LabelEncoder #encode target labels with value between 0 and n_classes-1
from sklearn.model_selection import train_test_split

data = pd.read_csv('car.data') #first 5 rows
#print(data.head())

X = data[['buying','maint','safety']].values
y = data[['class']]
#print(X,y)
#converting the data X
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:,i]= Le.fit_transform(X[:,i])
#print(X)

#y
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class']=y['class'].map(label_mapping)
y = np.array(y)
#print(y)

#creat model

knn = neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

knn.fit(X_train,y_train) #fit the k-nearest neighbors classifier from the training dataset
prediction = knn.predict(X_test)
accurary = metrics.accuracy_score(y_test, prediction)
print("prediction: ",prediction)
print("accurary: ",accurary)

print('actual value:', y[20])
print('predicted value',knn.predict(X)[20])