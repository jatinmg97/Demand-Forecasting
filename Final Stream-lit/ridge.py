import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from feature_engine import variable_transformers as vt
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn import metrics


class ridge_regression_Model():
    def __init__(self, df):
        self.df = df
    def mean_absolute_percentage_error(self,y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # calculate the aic
    def calculate_aic(self,n, mse, num_params):
        aic = n * log(mse) + 2 * num_params
        return aic

    def ridge_model(self,feat,target):
        df = self.df.loc[:, ~self.df.columns.duplicated()]
        
        # Separate into train and test sets

        X_train, X_test, y_train, y_test = train_test_split(df[feat],df[target], test_size=0.3, random_state=0)
      
        # calculate aic for regression
  
        X = X_train
        y = y_train
        # define and fit the model on all data
        model = Ridge()
        model.fit(X, y)
        intercept=model.intercept_
        print(intercept)
        intercept=intercept[0]
        # number of parameters
        num_params = len(model.coef_) + 1
        print('Number of parameters: %d' % (num_params))
        # predict the training set
        yhat = model.predict(X)
        # calculate the errors for train data
        mse = mean_squared_error(y, yhat)
        MSE_train = ('MSE for train data: %.3f' % mse)
        mae = (metrics.mean_absolute_error(y, yhat))  # For MAE
        MAE_train = ('MAE for train data: %.3f' % mae)
        rmse_train = (np.sqrt(metrics.mean_squared_error(y, yhat)))  # For RMSE
        RMSE_train = ('RMSE for train data: %.3f' % rmse_train)
        mape = self.mean_absolute_percentage_error(yhat, y)
        MAPE_train = ('MAPE for train data: %.3f' % mape)
        r2 = r2_score(y, yhat)
        R2_train = ('R2 for train data: %.3f' % r2)
        y_true, y_pred = np.array(y), np.array(yhat)
        idk= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
     

        aic = self.calculate_aic(len(y), mse, num_params)
        AIC_train = ('AIC for train data: %.3f' % aic)
        # calculate akaike information criterion for a linear regression model Test Data
        yhat = model.predict(X_test)
        # calculate the errors for test data
        mse = mean_squared_error(y_test, yhat)
        MSE_test = ('MSE for test data: %.3f' % mse)
        mae = (metrics.mean_absolute_error(y_test, yhat))  # For MAE
        MAE_test = ('MAE for test data: %.3f' % mae)
        rmse = (np.sqrt(metrics.mean_squared_error(y_test, yhat)))  # For RMSE
        RMSE_test = ('RMSE for test data: %.3f' % rmse)
        r2 = r2_score(y_test, yhat)
        R2_test = ('R2 for test data: %.3f' % r2)
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        idk2= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = self.mean_absolute_percentage_error(y_test,yhat )
        MAPE_test = ('MAPE for test data: %.3f' % mape)
        # calculate the aic
        aic = self.calculate_aic(len(y_test), mse, num_params)
        AIC_test = ('AIC for test data: %.3f' % aic)
        column = []
        names = []
        score = []
        name = []
        rfe = RFE(model)
        rfe.fit(X, y)
        test_t= X_test
        # summarize all features
        coefficients = dict()
        for feature,coef in zip(test_t.columns,model.coef_[0]):
            print(feature,coef)
            coefficients[feature]  = coef
        coefficients['intercept'] = intercept
       
        return (MSE_train, MAE_train, RMSE_train, R2_train, AIC_train,MAPE_train, MSE_test, MAE_test, RMSE_test, R2_test, AIC_test,MAPE_test,coefficients)
