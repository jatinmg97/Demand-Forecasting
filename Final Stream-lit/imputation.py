import pandas as pd
import numpy as np
import seaborn as sns
import pylab
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#from sklearn.impute import KNNImputer
from scipy import stats
from scipy.stats import shapiro
from datetime import *
from scipy import stats
from scipy.stats import skew
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import matplotlib.pyplot as plt
import joblib 
from pickle import dump

class imp:
    def __init__(self, data):
        self.data = data

    def summary(self):
        out = self.data.describe()
        return(out)

    def missing(self):
        percent_missing = self.data.isnull().sum() * 100 / len(self.data)
        missing_value_df = pd.DataFrame({'column_name': self.data.columns,
                                         'percent_missing': percent_missing})
        # print(missing_value_df.reset_index(drop=True))
        return missing_value_df.reset_index(drop=True)

    def skew(self):
        df_skew = self.data.skew()
        return df_skew

    def remove_duplicates(self):
        data = self.data
        self.df = data.loc[:, ~data.columns.duplicated()]
        return self.df

    def remove_null(self):
        remove_nan = self.data.replace(np.nan, 0)
        #drop = remove_nan.drop(['Date'], axis=1)
        return remove_nan

    def external(self, col_list):
        col_list1 = col_list.get('external')
        
        data = self.data[self.data.columns.intersection(col_list1)]
        z = np.abs(stats.zscore(data))
        threshold = 3
        a = np.where(z > 3)
        if len(a) >= 5:
            strat = "median"
        else:
            strat = "mean"
        print("Applying "+strat)
        imp_mean = SimpleImputer(strategy=strat)
        imp_mean.fit(data)
        data_mean = pd.DataFrame(imp_mean.transform(
            data), columns=data.columns, index=data.index)
        data_mean.to_csv('external.csv', index=False)
        return data_mean

    def internal(self, col_list):
        col_list1 = col_list.get('internal')
        data = self.data[self.data.columns.intersection(col_list1)]
        imp_mean = IterativeImputer(random_state=0)
        imp_mean.fit(data)
        data_iterative = pd.DataFrame(imp_mean.transform(
            data), columns=data.columns, index=data.index)
        data_iterative.to_csv('internal.csv', index=False)
        return data_iterative

    def granularity(self, col):
        data = self.data
        a = data[col]
        out = a.unique()
        print(out)
        return out

    def save(self):
        df = pd.read_csv(internal.csv)
        df2 = pd.read_csv(external.csv)

        df.to_csv()
        print("data saved as a csv")

    def aggregation(self, col):
        data = self.data
        myword = [col]
        m = data.isin(myword).any()
        cols = m.index[m].tolist()
        colu = cols[0]
        self.data_agg = data.loc[data[colu] == col]

        return self.data_agg

    def get_ts(self,data):
        df = data

        for col in df.columns:

            if df[col].dtype == 'object':
                try:
                    # print(col)
                    df[col] = pd.to_datetime(df[col])  # ,format="%d/%m/%Y")
                    # df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                    print(col)

                    self.df_sorted = df.sort_values(by=col)
                    print(df)
                    d1 = df[col].iloc[0]
                    d2 = df[col].iloc[1]

                    delta = d2-d1
                    month = timedelta(days=30)
                    weekly = timedelta(days=7)
                    daily = timedelta(days=1)
                    if delta > month:
                        print("data is monthly")
                    elif delta == weekly:
                        print("data is weekly")
                    elif delta == daily:
                        print("data is daily")
                    return self.df_sorted
                except:
                    pass

    def sortdate(self, name, column):
        data = self.data
        col = [str(column)]
        print(col)
        if name == "daily":
            return data
        elif name == "weekly":
            df = data.sort_values(by="Date")
            df.set_index('Date', inplace=True)
            df1 = df[col].resample('7D').first()
            df1 = df1.reset_index()
            return df1
        else:
            df = data.sort_values(by="Date")
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            print(df)
            df1 = df[col].resample('M').sum()
            df1 = df1.reset_index()
            return df1


class transformation:
    def __init__(self, drop):
        self.drop = drop

    def boxcox_entire(self):
        # generate non-normal data
        original_data = np.random.exponential(size=1000)

        # split into testing & training data
        train, test = train_test_split(original_data, shuffle=False)

        # transform training data & save lambda value
        train_data, fitted_lambda = stats.boxcox(train)

        # use lambda value to transform test data
        test_data = stats.boxcox(test, fitted_lambda)

        # (optional) plot train & test
        fig, ax = plt.subplots(1, 2)
        sns.distplot(train_data, ax=ax[0])
        sns.distplot(test_data, ax=ax[1])
        return fitted_lambda

    def skewed_boxcox(self):
        skwed_dist = stats.loggamma.rvs(5, size=10000) + 5
        skewed_box_cox, lmda = stats.boxcox(skwed_dist)
        sns.distplot(skewed_box_cox)
        stats.probplot(skewed_box_cox, dist=stats.norm, plot=pylab)
        pylab.show()
        result = ("lambda parameter for Box-Cox Transformation is:", lmda)
        return result

    # Standardization:
    def standardization(self):
        print("dsfsvf")
        print(self.drop)
        names = self.drop.columns
        # Create the Scaler object
        scaler = preprocessing.StandardScaler()
        # Fit your data on the scaler object
        scaler=scaler.fit(self.drop)
        scaled_df = scaler.transform(self.drop)
        dump(scaler, open('scaler.pkl','wb')) 
        print("mean")
        print(scaler.mean_)
        
        scaled_df = pd.DataFrame(scaled_df, columns=names)
        return [scaler, scaled_df]

    def convert_binary(self):
        x = self.drop.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x) 
        df = pd.DataFrame(x_scaled)
        return df


    def normalization(self,data):

        names=data.columns
   
        scaler=preprocessing.MinMaxScaler()
        scaler=scaler.fit(self.drop)
        normalized_df=scaler.transform(self.drop)
        dump(scaler, open('scaler2.pkl','wb'))
        
        normalized_df=pd.DataFrame(normalized_df,columns=names)
        return normalized_df

class transformation2:
	def __init__(self,dataset):
		self.dataset = dataset
	def trans(self):
        #Remove Dupliactes
		df = self.dataset.loc[:,~self.dataset.columns.duplicated()]
        #Remove Nan Values
		remove_nan = self.dataset.replace(np.nan,0)
        #Check For The DataTypes
		check_datatypes = remove_nan.dtypes
        #To Get The Column Date Column Number and Name
		Total_cloumns= remove_nan.columns
		if 'Date' or 'Timestamp' in remove_nan.columns:
			column_num = remove_nan.columns.get_loc("Date" or "Timestamp")
		else:
			print("no")
		date_column = Total_cloumns[column_num]
        #To Drop Date Column
		drop = remove_nan.drop([date_column], axis = 1)
        #To Describe Entire Dataset
		describe = drop.describe()
        #To Pre=Process The Dataset
		centered_scaled_data = preprocessing.scale(drop)
        #To Convert 1d-array
		dfconvert_array = drop.to_numpy()
		y = sum(dfconvert_array.tolist(),[])
        #Check If their are any negative or zero values
		neg_count = len(list(filter(lambda x: (x < 0), y))) 
		pos_count = len(list(filter(lambda x: (x > 0), y)))
		zero_count = len(list(filter(lambda x: (x <= 0), y)))
        #Plot Before Yeo-Johnson Transformation
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		prob = stats.probplot(y, dist=stats.norm, plot=ax1)
		ax1.set_xlabel('')
		ax1.set_title('Probplot against normal distribution')
        #Plot After Yeo-Johnson Transformation
		fig = plt.figure()
		ax2 = fig.add_subplot(212)
		xt, lmbda = stats.yeojohnson(y)
		prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
		ax2.set_title('Probplot after Yeo-Johnson transformation')
        #Skewness Before Transformation
		skewness_before = skew(drop)
        #Skewness After Transformation
		skewness_after = skew(xt)
        #Standardization
		names = drop.columns
		scaler = preprocessing.StandardScaler()
		scaled_df = scaler.fit_transform(drop)
		scaled_df = pd.DataFrame(scaled_df, columns=names)
        #Normalization
		x = drop.values
		min_max_scaler = preprocessing.MinMaxScaler()
		x_scaled = min_max_scaler.fit_transform(x)
		df = pd.DataFrame(x_scaled)
		return (df,remove_nan,check_datatypes,drop,describe,centered_scaled_data,lmbda,skewness_before,skewness_after,df)

'''

data=pd.read_csv('melted.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data[data.store != "NORTH_REGION"]
data = data[data.store != "SOUTH_REGION"]
print(data)
for index,row in data.iterrows():
    if row["store"]=="NORTH_STORE1" or "NORTH_STORE2":
        data['Region']="NORTH_REGION"
for index,row in data.iterrows():
    if row["store"]=="SOUTH_STORE1" or "SOUTH_STORE2" or "SOUTH_STORE3":
        data['Region']="SOUTH_REGION"
print(data)
print(data.dtypes)

data.loc[data['store'] == "NORTH_STORE1", 'Region'] = "NORTH_REGION"
data.loc[data['store'] == "NORTH_STORE2", 'Region'] = "NORTH_REGION"
print(data)
data.to_csv('melted2.csv',index=False)


df=data.sort_values(by="Date")
df.set_index('Date', inplace=True)
print(df)
df1=df.NORTH_STORE2.resample('M').first()
print(df1)



d1=df["Date"].iloc[0]
d2=df["Date"].iloc[1]
print(d2-d1)


date=data['Date']
data=data.drop(['Date'], axis=1)

imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(data)
data_iterative=pd.DataFrame(imp_mean.transform(data),columns=data.columns,index=data.index)
data_iterative['Date']=date
data_iterative.to_csv('imputed_data.csv',index=False)

data=pd.read_csv('imputed_data.csv')

new=data.melt(id_vars=['Year', 'Month', 'Date', 'Monthly_sales_adjusted_to_a_constant',
       'Sales_Scale_Constant', 'Monthly_sales_total', 'Market__Share',
       'Seasonality_.Monthly.', 'Trend', 'Random', 'Time_Variable',
       'Monthly_sales_lag1', 'Monthly_sales_lag2',
       'Expected_consumers_income_millions', 'Health_Consciousness_Rate',
       'Sulphate_content', 'Personal_Consumption_Expenditures',
       'Population_growth', 'Consumer_Price_Index_Urban',
       'Total_Market_Size_Cumulative', 'Total_Market_Size_Monthly',
       'Market_Share', 'Price_Per_Unit', 'Advertising_costs',
       'Advertising_costs_lag1', 'Advertising_costs_lag2',
       'Advertising_costs_lag3', 'Advertising_costs_lag4',
       'Advertising_costs_lag5', 'Advertising_costs_lag6',
       'Advertising_costs_lag7', 'Advertising_costs_lag8',
       'Advertising_costs_lag9', 'Number_of_Distributors',
       'Number_of_Retail_Locations', 'Discount_Percentage', 'Temperature'])

new=new.rename(columns={"variable": "store", "value": "Sales"})
new.to_csv('melted.csv',index=False)




# check missing percent
percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_df = pd.DataFrame({'column_name': data.columns,
                                 'percent_missing': percent_missing})
print(missing_value_df.reset_index(drop=True))


# simple imputer for mean mode median
imp_mean = SimpleImputer(strategy='mean')
imp_mean.fit(data)
data_mean=pd.DataFrame(imp_mean.transform(data),columns=data.columns,index=data.index)
print(data_mean)

percent_missing = data_mean.isnull().sum() * 100 / len(data_mean)
missing_value_df = pd.DataFrame({'column_name': data_mean.columns,
                                 'percent_missing': percent_missing})
print(missing_value_df.reset_index(drop=True))


# sklearn iterative imputer method
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(data)
data_iterative=pd.DataFrame(imp_mean.transform(data),columns=data.columns,index=data.index)
print(data_iterative)

# sklearn knn imputer

imp_mean = KNNImputer(n_neighbors=2)
imp_mean.fit(data)
data_knn=pd.DataFrame(imp_mean.transform(data),columns=data.columns,index=data.index)
print(data_knn)


# multiple imputation
# data_matrix=data.to_numpy()
# data_multiple=pd.DataFrame(data=mice.complete(data_matrix), columns=data.columns, index=data.index)
# print(data_multiple)


data=pd.read_csv('Product_data.csv')
stat, p = shapiro(data['Month'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
'''
