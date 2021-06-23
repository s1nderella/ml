from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

bc = load_breast_cancer()
bc_df = pd.DataFrame(bc.data, columns = bc.feature_names)
print(bc_df.head())
x = bc.data
y = bc.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=6)

model = KMeans(n_clusters=2,random_state=0)
model.fit(x_train)
predictions = model.predict(x_test)
labels = model.labels_
print('labels: ',labels)
print('prediction: ',predictions)
print('accuracy: ',accuracy_score(y_test,predictions))
print('actual: ',y_test)
