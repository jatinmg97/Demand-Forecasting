import pandas as pd 
import numpy as np

data=pd.read_csv('Product_data.csv')

data['Cost_Per_Unit']=0.3*data["Price_Per_Unit"]
print(data)
data.to_csv("Product_data.csv",index=False)
