# decision tree for feature importance on a regression problem
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from boruta import BorutaPy
from imputation import imp , transformation ,transformation2
import numpy as np
# define dataset
data=pd.read_csv("melted2.csv")
data=data.loc[data['store'] == "NORTH_STORE1"]
data=data.drop('Date',axis=1)
data=data.drop("store",axis=1)
data=data.drop('Region',axis=1)

data=data.drop('Monthly_sales_total',axis=1)
from sklearn.preprocessing import StandardScaler
names=data.columns
scaler = StandardScaler()
print(scaler.fit(data))
scaled_df=scaler.transform(data)

data= pd.DataFrame(scaled_df, columns=names)
print(data)
y=data['Sales']
data=data.drop("Sales",axis=1)
X=data.iloc[:, 0:37]
print(X.columns)
print(X)
features=list(X.columns)

rf = RandomForestRegressor(n_estimators = 100, n_jobs=-1, oob_score=True)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(X.values, y.values)

# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
print(feat_selector.ranking_)

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X.values)


final_features = list()
indexes = np.where(feat_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(features[x])
print(final_features)





# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_



# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
