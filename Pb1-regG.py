'''
Group 43
Gabriel Marques - 93067
Manuel Miranda - 93124
'''
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.linear_model import Lasso

#Load X_train file into numpy matrix
raw_data_X = np.load("Xtrain_Regression_Part1.npy")
#Load y_train file into numpy matrix
raw_data_Y = np.load("Ytrain_Regression_Part1.npy")
#Load X_test file into numpy matrix
X_test =np.load("Xtest_Regression_Part1.npy")

raw_data_Y=raw_data_Y.ravel()

#Finding the best alpha for the model (in this case alpha = 0.0014749266261809002)
modelo_lassoCV = LassoCV(cv=10, fit_intercept=True,tol=0.001,eps=0.0000001,n_alphas=95).fit(raw_data_X, raw_data_Y)
#Creation of the lasso model to predict y
print(modelo_lassoCV.alpha_)
model_lasso = Lasso(alpha = modelo_lassoCV.alpha_,tol=0.001,fit_intercept=True)
#Fit the model with the train data
model_lasso.fit(raw_data_X,raw_data_Y)

y_pred = model_lasso.predict(X_test) #predict the awnser for the X_test

np.save("y_predictions",y_pred) #saving the the predictions of y into a .npy file