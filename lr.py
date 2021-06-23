from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
boston = datasets.load_boston()
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_df['MEDV'] = boston.target

boston_df.head()

X = boston.data
y = boston.target
'''print('X')
print(X)
print(X.shape)
print('y')
print(y)
print(y.shape)'''

                 
#algorithm
l_reg = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# train
model = l_reg.fit(X_train,y_train)
pred = model.predict(X_test)
train_pred = model.predict(X_train)

print("Prediction: ", pred)
print("R^2 value: ", l_reg.score(X,y))
print("coeff: ", l_reg.coef_)
print("intercept: ", l_reg.intercept_)

# mean square error
print('MSE train data: ', mean_squared_error(y_train, train_pred)) 
print('MSE test data: ', mean_squared_error(y_test, pred))

# graph plotting
plt.scatter(pred, pred - y_test, color = 'blue')      
plt.hlines(y = 0, xmin = -10, xmax = 50, color = 'black') 
plt.title('Residual Plot')                                
plt.xlabel('Predicted Values')                            
plt.ylabel('Residuals')                                   
plt.grid()                                                
plt.show()  