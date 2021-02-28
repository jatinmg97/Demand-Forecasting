from imputation import imp , transformation ,transformation2
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

data=pd.read_csv('latest_data.csv')
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

data=data.iloc[:,1:]
print(data.head())

model = VAR(data)
model_fit = model.fit()

pred = model_fit.forecast(model_fit.y, steps=1)
print(pred)

# Augmented Dickey-Fuller Test (ADF Test)/unit root test
from statsmodels.tsa.stattools import adfuller
def adf_test(ts, signif=0.05):
    dftest = adfuller(ts, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])
    for key,value in dftest[4].items():
       adf['Critical Value (%s)'%key] = value
    print (adf)
    
    p = adf['p-value']
    if p <= signif:
        print(f" Series is Stationary")
    else:
        print(f" Series is Non-Stationary")

#apply adf test on the series
adf_test(data["NORTH_STORE1"])

df_differenced = data.diff().dropna()
# stationarity test again with differenced data
adf_test(df_differenced["NORTH_STORE1"])

'''
imp = transformation2(data)
transformations = imp.trans()
print(transformations)


#input the entire dataset
somevar=imp(data)

#get the summary of dataset
print(somevar.summary())

#get the skew value
print(somevar.skew())

#get the missing value percentage
print(somevar.missing())

remove_dup = somevar.remove_duplicates()
print(remove_dup)

drop = somevar.remove_null()
print(drop)

col=list(data.columns)

variabledict={"external":["Health_Consciousness_Rate","Temperature"],"internal":["Price_Per_Unit","Sulphate_content"]}

external_var=["Health_Consciousness_Rate","Temperature"]

#pass external variable dict
out=somevar.external(variabledict)
print(out)


internal_var=["Price_Per_Unit","Sulphate_content"]



#pass internal variable dict
out=somevar.internal(variabledict)
print(out)


imp_data=pd.read_csv('melted2.csv')

trans=imp(imp_data)

#select the column you want to aggregate
trans.granularity('store')

aggregated_data=trans.aggregation('NORTH_STORE1')

#print(aggregated_data)

#Checks the time series data(daily , weekly , monthly)

sorted_data=trans.get_ts()

#sorted_data.set_index('Date', inplace=True)
dt=sorted_data['Date']
sorted_data=sorted_data.drop(['store', 'Region','Date'], axis=1)

print(sorted_data)
#### transformation
a = transformation(sorted_data)
skewed = a.skewed_boxcox()
print(skewed)

#standardization
standrd = a.standardization()
print(standrd)
standrd=standrd[1]
standrd['Date']=dt
print(standrd.dtypes)

#normalization
normalize = a.normaliz_entire()
normalize=normalize[1]
print(normalize)
binary = a.convert_binary()
print(binary)





#pass the breakdown and the target variable
data=imp(standrd)
out=trans.sortdate("monthly","Sales")


print(out)
print(out.dtypes)




'''