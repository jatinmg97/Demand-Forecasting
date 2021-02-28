from arimamodel import *
import pandas as pd
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm 
from pandas import read_csv
import numpy as np
from pycaret.datasets import get_data
from pycaret.regression import *
import seaborn as sns
'''
df = pd.read_csvdata=pd.read_csv('kohlerdataset.csv')

arima_model1 = Arima_Model(df)
arima_results = arima_model1.arima_model()
print(arima_results)
'''
data=pd.read_csv("Product_data.csv")

corr = data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


