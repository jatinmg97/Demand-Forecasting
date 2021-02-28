from linear_regression import *
import pandas as pd

dataset = pd.read_csv("latest_data.csv", sep=",")

linear_model1 = linear_regression_Model(dataset,feat,target)
linear_results = linear_model1.linear_model()
st.write(linear_results[0])
