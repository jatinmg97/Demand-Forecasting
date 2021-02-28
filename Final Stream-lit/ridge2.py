from ridge import *
import pandas as pd

dataset = pd.read_csv("latest_data.csv", sep=",")

#linear_model1 = ridge_regression_Model(dataset)
#linear_results = linear_model1.ridge_model(["Temperature","Price_Per_Unit"],"NORTH_STORE1")

from scipy.optimize import linprog
c=[-0.4144 ,0.1197]
x0_bounds = (2, 3)
x1_bounds = (-3, 1)
res = linprog(c, bounds=[x0_bounds, x1_bounds])
print(res)
