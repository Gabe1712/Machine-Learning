'''
Group 43
Gabriel Marques - 93067
Manuel Miranda - 93124
'''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, Lasso

#Load X_train file into numpy matrix
raw_data_X = np.load("Xtrain_Regression_Part2.npy")
#Load y_train file into numpy matrix
raw_data_Y = np.load("Ytrain_Regression_Part2.npy")
#Load X_test file into numpy matrix
X_test =np.load("Xtest_Regression_Part2.npy")

raw_data_Y=raw_data_Y.ravel()

#We made a linear regression so that we can see witch points will be the outliers
model_lr = LinearRegression()
model_lr.fit(raw_data_X,raw_data_Y)
y_p=model_lr.predict(raw_data_X)

error=raw_data_Y-y_p

#After we obtain the error matrix, we will find the 2 point with worst error
ind = np.argsort(np.abs(error))[-2:]


#Values of X_train and Y_train without the outliers
newY=np.delete(raw_data_Y,ind)
newX=np.delete(raw_data_X,ind, axis=0)

#Finding the best alpha for the model 
modelo_lassoCV = LassoCV(cv=10).fit(newX, newY)

model_lasso = Lasso(alpha = modelo_lassoCV.alpha_)
model_lasso.fit(newX,newY)

y_pred = model_lasso.predict(X_test) #predict the awnser for the X_test

np.save("y_predictions_G43",y_pred) #saving the the predictions of y into a .npy file

y_try =np.load("y_predictions_G43.npy")
print(y_try)