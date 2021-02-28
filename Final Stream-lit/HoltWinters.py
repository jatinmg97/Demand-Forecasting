from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.validation import (array_like, bool_like, float_like,
                                          string_like, int_like)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from numpy import mean
class holt_winters():
    def __init__(self,endog,crossvalidation = 5, seasonal_periods = None, freq=None,damped=None):
        self.n_splits = crossvalidation
        #self.damped = bool_like(damped, 'damped')
        
        self.seasonal_periods =  7 
        self.endog = endog
        self.timeserieslength = len(self.endog)
        self.predictions = []

    def fit(self,use_boxcox=True):
        
        no_splits = self.n_splits
        trainsplit = TimeSeriesSplit(no_splits) 
        rmse = {'add_add':[],'mul_add':[],
                'add_mul':[],'mul_mul':[]}
        endog = self.endog
        for train,test in trainsplit.split(endog):
            self._data = endog[train]
            endog_test = endog[test]
            #Additive trend and seasaonality 
            
            forecast_period = len(test)
            seasonal_add_trend_add_model = ExponentialSmoothing(self._data,seasonal_periods=7, seasonal = 'add', trend = 'add').fit(use_boxcox = use_boxcox)
            print("data")
            print(endog_test)
            forecast=seasonal_add_trend_add_model.forecast(forecast_period)
            rmse['add_add']=mean_squared_error(seasonal_add_trend_add_model.forecast(forecast_period),endog_test)
            #rmse['add_add']=mean_squared_error(endog_test,forecast)

            #Multiplicative seasonality and additive trend
            seasonal_mul_trend_add_model = ExponentialSmoothing(self._data,seasonal_periods=7,trend='add',seasonal='mul').fit(use_boxcox = use_boxcox)
            rmse['mul_add'].extend
            (mean_squared_error(seasonal_mul_trend_add_model
            .forecast(forecast_period),endog_test)) 

            #Multiplicative seasonality and trend
            seasonal_mul_trend_mul_model = ExponentialSmoothing(self._data,seasonal_periods=7,trend='mul',seasonal='add').fit(use_boxcox = use_boxcox)
            print(seasonal_mul_trend_mul_model.forecast(forecast_period))
            rmse['mul_mul']=mean_squared_error(seasonal_mul_trend_mul_model
            .forecast(forecast_period),endog_test)

             #Additive seasonality and multiplicative trend
            seasonal_add_trend_mul_model =ExponentialSmoothing(self._data,seasonal_periods=7,trend='add',seasonal='add').fit(use_boxcox = use_boxcox)
            rmse['add_add']=mean_squared_error(seasonal_add_trend_mul_model
            .forecast(forecast_period),endog_test)

        #replace the self.endog with the actual data
        self.endog = endog
        cv_mean_rmse = {k:mean(v) for k,v in rmse.items()}
        min_rmse = min(cv_mean_rmse,key  =cv_mean_rmse.get).split('_') #extract the seasonal and trend that gave least rmse
        
        
        return ExponentialSmoothing(self._data,seasonal_periods=7,trend=min_rmse[1],seasonal=min_rmse[0]).fit(use_boxcox = use_boxcox)

    def metrics(self):
        #mae = mean_absolute_error(self.forecast(30),test)
        #rmse  = mean_squared_error(self.fit.forecast(30),test)
        #m = self.timeserieslength/self.seasonal_periods
        return None