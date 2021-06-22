import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets,svm
from sklearn.preprocessing import LabelEncoder #encode target labels with value between 0 and n_classes-1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
#split it in features and labels
X = iris.data
y = iris.target

classes = ['Iris Setosa','Iris Versicolour','Iris Virginica']
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = svm.SVC()
model.fit(X_train,y_train)

print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test,predictions)
print("prediction: ",predictions)
print("actual: ",y_test)
print("accurary: ",acc)

"""for i in range(len(predictions)):
    print(classes[predictions[i]])"""

