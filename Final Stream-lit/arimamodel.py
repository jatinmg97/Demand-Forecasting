import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import acf,pacf
from datetime import datetime, timedelta  
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
import streamlit as st
rcParams['figure.figsize'] = 10,6

data=pd.read_csv(r'Product_data.csv')

class Arima_Model():
	def __init__(self,df):
		self.df = df
		
	def arima_model(self):
		# To Identify No.of Rows In a DataSet
		no_rows = len(self.df)

		#To Assign DataSet Column Names to Variables 
		Total_cloumns= self.df.columns
		inp_column = Total_cloumns[0]
		out_column = Total_cloumns[1]

		#To Index The Date Column
		inp_col = pd.to_datetime(self.df[inp_column])
		Dataset = self.df.set_index([inp_column])

		#To Split The Dataset Into Train & Test
		x = self.df[inp_column]
		y = self.df[out_column]
		tscv = TimeSeriesSplit()            
		TimeSeriesSplit(max_train_size=None, n_splits=5)
		for train_index, test_index in tscv.split(Dataset):
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]

		#To Make Train DataSet
		train_timestamp = pd.DataFrame(x_train)
		train_timestamp_1= train_timestamp.rename(columns = {0:inp_column},inplace = True)
		train_usedspace = pd.DataFrame(y_train)
		train_usedspace_1= train_timestamp.rename(columns = {0:out_column},inplace = True)
		trdata = pd.concat([train_timestamp, train_usedspace],axis = 1)
		trdata = trdata.set_index([inp_column])

		#To Identify No.of Rows For Y-Train
		rows_ytrain = len(y_train)

		#To Identify No.of Rows For Y-Test
		rows_ytest = len(y_test)

		#To Describe The Entire DataSet
		dataset_des = Dataset.describe()
		#print(dataset_des)

		#To Describe The Training DataSet
		trdata_des = trdata.describe()
		

		#Apply Log Transform For Training DataSet & Remove NaN Values
		Dataset_logScale = np.log(trdata)
		
		datasetLogDiffshifting=Dataset_logScale-Dataset_logScale.shift()
		datasetLogDiffshifting.dropna(inplace=True)
		#Dataset_logScale=trdata

		#components of time-series(Plot Trend, Seasonal And Residuals)
		decomposition=seasonal_decompose(Dataset_logScale,period=3)
		trend=decomposition.trend
		seasonal=decomposition.seasonal
		residual=decomposition.resid
		plt.show()
		plt.plot(Dataset_logScale, label='Original')
		plt.legend(loc='best')
		plt.subplot(412)
		plt.plot(trend,label='Trend')
		plt.legend(loc='best')
		plt.subplot(413)
		plt.plot(seasonal,label='Seasonality')
		plt.legend(loc='best')
		plt.subplot(414)
		plt.plot(residual,label='Residuals')
		plt.legend(loc='best')
		plt.tight_layout()
		decomposedLogData=residual
		decomposedLogData.dropna(inplace=True)

		#To Identify Variance
		seasonal_max = seasonal.max()
		seasonal_min = seasonal.min()
		trend_max = trend.max()
		trend_min = trend.min()
		variance = (seasonal_max-seasonal_min)/(trend_max-trend_min)*100

		#To Find Variance Of Seasonal
		variance_seasonal= np.var(seasonal)

		#To Find Variance Of Trend
		variance_trend=np.var(trend)

		#To Find Variance Of Residuals
		variance_residual=np.var(residual)

		#Auto co-relation and Partial Auto Co-relation Functions
		lag_acf = acf(datasetLogDiffshifting, nlags=20)
		lag_pacf = pacf(datasetLogDiffshifting, nlags=20, method='ols')    #ols=ordinary least square method
		
		#plot ACF:
		plt.subplot(121)
		plt.plot(lag_acf)
		plt.axhline(y=0,linestyle='--', color='gray')
		plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='--',color='gray')
		plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='--',color='gray')
		plt.title('AutoCorrelation Function')
		st.pyplot()
		#plot PACF:
		plt.subplot(122)
		plt.plot(lag_pacf)
		plt.axhline(y=0,linestyle='--', color='gray')
		plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='--',color='gray')
		plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='--',color='gray')
		plt.title('Partial AutoCorrelation Function')
		plt.tight_layout()
		st.pyplot()
		
		#Apply AR Model:
		print(Dataset_logScale)
		model = ARIMA(Dataset_logScale, order=(3,1,3))
		results_AR= model.fit(disp=-1)
		plt.plot(datasetLogDiffshifting)
		plt.plot(results_AR.fittedvalues, color='red')   #residual sum of square
		plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffshifting[out_column])**2))

		#Apply MA Model:
		model=ARIMA(Dataset_logScale,order=(3,1,3)) #moving Average Model
		results_MA= model.fit(disp=-1)
		plt.plot(datasetLogDiffshifting)
		plt.plot(results_MA.fittedvalues, color='red')
		plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffshifting[out_column])**2))
		st.pyplot()
		#Integrate Both As ARIMA Model:
		model=ARIMA(Dataset_logScale,order=(3,1,3)) #plotting for ARIMA 
		results_ARIMA= model.fit(disp=-1)
		plt.plot(datasetLogDiffshifting)
		plt.plot(results_ARIMA.fittedvalues, color='red')
		plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffshifting[out_column])**2))
		st.pyplot()
		#Fitting ARIMA Model And Converting Cumulative Sum
		predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues, copy=True) #fitting ARIMA model
		predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum() #convereted to cumulative sum
		predictions_ARIMA_log= pd.Series(Dataset_logScale[out_column].iloc[0],index=Dataset_logScale.index)
		predictions_ARIMA_log= predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

		#Predictions Of ARIMA Model
		pred = results_ARIMA.predict(start=1,end=rows_ytrain)
		predictions_ARIMA_diff = pd.Series(pred, copy=True)
		predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
		predictions_ARIMA_log = pd.Series(Dataset_logScale.iloc[0])
		predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
		predictions_ARIMA = np.exp(predictions_ARIMA_log)
		s=pd.DataFrame(predictions_ARIMA)
		s=s.reset_index()

		#Forecasted Plot For ARIMA Model
		st.write("Forecasted Plot For ARIMA Model")
		p=results_ARIMA.plot_predict(1,550)
		plt.xlabel('Timestamp',fontsize=14, color='b')
		plt.ylabel('Sales',fontsize=14,color='b')
		plt.title('Forecast ',fontsize=20,color='black')
		plt.legend(['Forecasted Data','Input Data'], loc ='upper left')
		plt.axhline(y=1000, color='r', linestyle='-')
		plt.ylim(0,12)
		plt.show()
		st.pyplot()

		#Forecasted Results Of Y-test For ARIMA Model
		forecast = results_ARIMA.forecast(steps=rows_ytest)[0]
		y_pred = (forecast*100)/2

		from sklearn import metrics
		MAE = (metrics.mean_absolute_error(y_test,y_pred)) #To Find MAE Value

		MSE = (metrics.mean_squared_error(y_test,y_pred)) #To Find MSE Value

		RMSE = (np.sqrt(metrics.mean_squared_error(y_test,y_pred))) #To Find RMSE Value

		def mean_absolute_percentage_error(y_true, y_pred):  #To Find MAPE Value
			y_true, y_pred = np.array(y_true), np.array(y_pred)
			return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
		mape=mean_absolute_percentage_error(y_test,y_pred)

		from sklearn.metrics import r2_score  #To Find R2 Value
		r2 = r2_score(y_test, y_pred)

		#To Make DataFrame and Rename For Forecasted log Results Of Y-Test
		s= s.drop(columns= ["index"])
		s.rename(columns = {0:out_column},inplace = True)

		#To Find Fitted Values
		train_y = y_train.to_frame()
		train_y = np.log(train_y)
		fitted_values = ((train_y-s)/train_y)*100
		fitted_values_1 = fitted_values.rename(columns = {out_column:"Fitted_values"},inplace = True)

		#To Find Predicted Values
		predicted_values = pd.DataFrame(y_pred)
		predicted_values1 = predicted_values.rename(columns = {0:"Predicted_values"},inplace = True)

		#To Find Forecasted Values For A Quarter-Period
		quarter_period = 90
		forecasted_days = rows_ytest + quarter_period
		forecast = results_ARIMA.forecast(steps=forecasted_days)[0]
		for_val = (forecast*100)/2

		#Assign Name To Forecasetd Values and Make into DataFrame
		forecast = pd.DataFrame(for_val)
		forecast.rename(columns = {0:"Forecasted_values"},inplace = True)
		forecast=forecast.iloc[50:]
		
		#Convert all DataFrames into Numpy.array
		fitted_values = fitted_values.Fitted_values.to_numpy()
		predicted_values = predicted_values.Predicted_values.to_numpy()
		forecasted_values = forecast.Forecasted_values.to_numpy()
		timestamp = self.df.Date.to_numpy()
		actual_values = data.Monthly_sales_total.to_numpy()

		#To Make Alignment For Final Report
		length = len(fitted_values)
		an_array = np.empty(length)
		an_array[:] = 0
		final_predicted = np.concatenate((an_array, predicted_values))
		length1 = len(actual_values)
		an_array1 = np.empty(length1)
		an_array1[:] = 0
		final_forecasted = np.concatenate((an_array1, forecasted_values))

		#To Make All Results Into The DataFrame
		dict = {'Date': timestamp, 'Actual_values': actual_values, 'Fitted_values': fitted_values, 'Predicted_values': final_predicted, 'Forecasted_values': final_forecasted} 
		df = pd.DataFrame.from_dict(dict, orient='index')
		df.transpose()

		#To Generate CSV Report File
		df.to_csv(r'C:\Users\nkatakamsetty\Desktop\Demand_Forecast\final_report.csv',index = True) 
		return [trdata,dataset_des,trdata_des,variance,variance_seasonal,variance_trend,variance_residual,s,MAE,MSE,RMSE,mape,r2,df]
