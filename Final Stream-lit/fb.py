import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
'''
data=pd.read_csv('kohlerdataset.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])


data = data.rename(columns = {'Timestamp': 'ds',
                                'UsedSpace': 'y'})

data2 = data.copy()

data['new']=data.y.shift().rolling(3,min_periods=1).mean().fillna(data.y)


print(data)
#tranformation
tar=data['y']                              
val=tar.values
val=val.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MinMaxScaler()
#scaler.fit((val))
#transformed=scaler.transform(val)
#print(transformed)
#data['y']=transformed
data[['y', 'new']] = scaler.fit_transform(data[['y', 'new']])
print(len(data['y']))
#train, test = train_test_split(data, test_size=0.20, shuffle=False)
train=data.head(270)
test=data.tail(30)
#print(train)
print(test)






model = Prophet(interval_width = 0.95,weekly_seasonality=True,changepoint_prior_scale=0.01,seasonality_mode='multiplicative')
#model.add_seasonality(name='daily', period=30.5,fourier_order=5)
# model = Prophet(mcmc_samples=2)
model.add_regressor('new')

model.fit(train)


# dataframe that extends into future 6 weeks 
future_dates = model.make_future_dataframe(periods = 30)
future_dates['new']=data['new']

print("First week to forecast.")
print(future_dates.tail(5))

# predictions
forecast = model.predict(future_dates)

# preditions for last week
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))

fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})
pred=fc['yhat']     
fc["new"]=data["new"]
#inverse tranform
pred=pred.values
pred=pred.reshape(-1, 1)
#inversed = scaler.inverse_transform(pred)
fc[['yhat', 'new']] = scaler.inverse_transform(fc[['yhat', 'new']])
#print(inversed)
#fc['yhat']=inversed
fc1=fc.tail(30)
print(fc1)
'''
'''
actual=test['y']                              
actual=actual.values
actual=actual.reshape(-1, 1)
inversed = scaler.inverse_transform(actual)
test['y']=inversed
'''
'''
test=data2.tail(30)
print(test)




data=pd.read_csv('kohlerdataset.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

charts=data[['Timestamp','UsedSpace']]
charts['Predicted']=fc['yhat']
print(charts)

model.plot(forecast)

from sklearn.metrics import mean_squared_error
from math import sqrt

print("rms value")
rms = sqrt(mean_squared_error(test['y'],fc1['yhat'] ))

print(rms)

print("rsquared values")
from sklearn.metrics import r2_score
R2=r2_score(test['y'],fc1['yhat'] )
print(R2)

print("mae value")
mae=mean_absolute_error(test['y'],fc1['yhat'])
print(mae)


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape=mean_absolute_percentage_error(test['y'],fc1['yhat'])
print("mape value")
print(mape)




charts.set_index('Timestamp', inplace=True)
charts.plot(legend=True)

plt.show()

model.plot(forecast)

'''

data=pd.read_csv('melted2.csv')
data['Date'] = pd.to_datetime(data['Date'])
#print(data)

from imputation import *
trans=imp(data)

#select the column you want to aggregate
trans.granularity('store')

aggregated_data=trans.aggregation('NORTH_STORE1')


#regressor=aggregated_data[['Price_Per_Unit']]
print(aggregated_data)

#aggregated_data['new']=aggregated_data.Sales.shift().rolling(3,min_periods=1).mean().fillna(aggregated_data.Sales)
charts=aggregated_data[['Sales','Price_Per_Unit']]

scaler = MinMaxScaler()
charts[['Sales','Price_Per_Unit']] = scaler.fit_transform(charts[['Sales','Price_Per_Unit']])


import statsmodels.api as sm
from statsmodels.tsa.api import VAR

train=charts.head(150)
test=charts.tail(4)
print(train)
model = VAR(train)
model_fit = model.fit()

pred = model_fit.forecast(model_fit.y, steps=4)
pred=pd.DataFrame(pred, columns=['Sales','Price_Per_Unit']) 
print(pred)

pred[['Sales', 'Price_Per_Unit']] = scaler.inverse_transform(pred[['Sales', 'Price_Per_Unit']])

print(aggregated_data[['Sales','Price_Per_Unit']].tail(4))
print(pred)
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape=mean_absolute_percentage_error(test['Sales'],pred['Sales'])
print("mape value")
print(mape)


'''
tar=aggregated_data['Sales']                              
val=tar.values
val=val.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit((val))
transformed=scaler.transform(val)
print(transformed)
aggregated_data['Sales']=transformed
'''
'''





#train_r=regressor.head(150)
#test_r=regressor.tail(4)

aggregated_data=aggregated_data[['Date','Sales']]
data=aggregated_data.sort_values(by='Date')
data = data.rename(columns = {'Date': 'ds',
                                'Sales': 'y'})
print(data)

train=data.head(150)
test=data.tail(4)
print(test)


model = Prophet(interval_width = 0.95,weekly_seasonality=True,changepoint_prior_scale=0.01,seasonality_mode='multiplicative')
#model.add_regressor('new')
model.fit(train)


future_dates = model.make_future_dataframe(periods = 4,freq='M')
#future_dates['new']=aggregated_data["new"]

# future_dates['Price_Per_Unit']=aggregated_data['Price_Per_Unit']
print(future_dates)
forecast = model.predict(future_dates)
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(model, horizon = '120 days')
print(df_cv.head())
df_cv['yhat']=scaler.inverse_transform(df_cv[['yhat']])
df_cv['yhat_lower']=scaler.inverse_transform(df_cv[['yhat_lower']])
df_cv['yhat_upper']=scaler.inverse_transform(df_cv[['yhat_upper']])
df_cv['y']=scaler.inverse_transform(df_cv[['y']])
print(df_cv.head())

from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
print(df_p.head())

# preditions for last week
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))

fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})

pred=fc['yhat']                              

fc["new"]=aggregated_data["new"]
fc[['yhat', 'new']] = scaler.inverse_transform(fc[['yhat', 'new']])




fc1=fc.tail(4)
print(fc1)
print(test)

test[['y', 'new']] = scaler.inverse_transform(test[['y', 'new']])



print(test)
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.metrics import mean_squared_error
from math import sqrt

print("rms value")
rms = sqrt(mean_squared_error(test['y'],fc1['yhat'] ))

print(rms)

print("rsquared values")
from sklearn.metrics import r2_score
R2=r2_score(test['y'],fc1['yhat'] )
print(R2)

print("mae value")
mae=mean_absolute_error(test['y'],fc1['yhat'])
print(mae)


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape=mean_absolute_percentage_error(test['y'],fc1['yhat'])
print("mape value")
print(mape)

#print(len(inversed2))
#print(len(charts['Sales']))
charts['Predicted']=fc['yhat']
print(charts)

model.plot(forecast)

charts=charts.tail(24)
charts.set_index('Date', inplace=True)
charts.plot(legend=True)

plt.show()

model.plot(forecast)
'''

