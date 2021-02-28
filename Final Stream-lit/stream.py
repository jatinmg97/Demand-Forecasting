import streamlit as st
import random
import statsmodels.api as sm
import numpy as np
from sympy import *
#from fbprophet import Prophet
from imputation import *
import matplotlib.pyplot as plt
import joblib
from HoltWinters import holt_winters
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from arimamodel import *
from scipy.optimize import linprog
from sklearn.tree import DecisionTreeRegressor
from linear_regression import *
import pandas as pd
from ridge import *
from matplotlib import pyplot
#from boruta import BorutaPy
import time
from sklearn.linear_model import LinearRegression
from sklearn import tree
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_linear_regression
from pickle import load
import seaborn as sns
sns.set(color_codes=True)
st.title('Demand Forecasting')
st.markdown("Welcome to this Streamlit App")
page = st.sidebar.selectbox("Choose a page", ["Homepage", "Data Upload", "Data-Statistics", "Data-Imputation",
                                              "Data-Transformation", "Uni-Variate Analysis", "Bi-Variate Analysis", "Multi-Variate Analysis","Scenario Analysis","Optimum Variable Value Selection"])
st.sidebar.warning("Please follow the sequence")
data = pd.read_csv('Product_data.csv')
# data.to_csv('latest_data.csv',index=False)
if page == "Homepage":
    st.subheader('This is the homepage !')
elif page == "Data Upload":
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("This is your uploaded data")
        st.table(data)

    data.to_csv('data.csv')

elif page == "Data-Statistics":
    st.markdown(" This is the Data statistics page")
    data_load_state = st.text('Loading data...')
    data = pd.read_csv('Product_data.csv')
    somevar = imp(data)
    data_load_state.text('Loading data...done!')

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.table(data.head())

    st.subheader('Summary of Data')
    # get the summary of dataset
    a = somevar.summary()
    st.write(a)
    st.subheader('Missing Data')
    b = somevar.missing()
    st.write(b)
    st.subheader('Charts/EDA')
    column_name = st.selectbox("Select column for EDA ", data.columns.tolist())
    st.write('You selected:', column_name)
    st.bar_chart(data[column_name])

elif page == "Data-Imputation":
    st.title("Steps for pre processing")
    if st.checkbox("1. Remove duplicates"):
        somevar = imp(data)
        remove_dup = somevar.remove_duplicates()
        remove_dup.to_csv("latest_data.csv", index=False)
        st.markdown("Duplicates removed")
        st.dataframe(remove_dup)
    st.header("Please identify the internal and external variables")
    data = pd.read_csv('latest_data.csv')
    date = data['Date']
    col = list(data.columns)
    st.warning(
        "Please select carefully, as they will decide the data imputation techniques")
    internal = st.multiselect("Internal var", col)
    st.write("You selected", len(internal), "Internal Variables")
    # col=list(set(col)^set(internal))
    external = st.multiselect("External var", col)
    st.write("You selected", len(external), "External Variables")
    feat = internal+external
    joblib.dump(internal,'internal.pkl')
    joblib.dump(external,'external.pkl')
    joblib.dump(feat, "features.pkl")
    st.header("Please select the Target Variable")
    target = str(st.selectbox("Target var", col))
    joblib.dump(target, "target.pkl")
    submit = st.button("Submit")
    if submit:
        external.append(target)
        variabledict = {"external": external, "internal": internal}
        somevar = imp(data)
        intvar = somevar.internal(variabledict)
        extvar = somevar.external(variabledict)
        imputed_data = pd.concat([intvar, extvar], axis=1)
        # joblib.dump(features,"features.pkl")
        imputed_data.insert(loc=0, column='Date', value=date)
        imputed_data.to_csv("imputed_data.csv", index=False)
        imputed_data.to_csv("latest_data.csv", index=False)
        st.success("Data-Imputation done on the selected variables")
        st.text("Imputed data")
        st.write(imputed_data)

elif page == "Data-Transformation":
    st.header("Data-Tranformation")
    data = pd.read_csv('latest_data.csv')
    date = data['Date']
    data = data.drop(["Date"], axis=1)
    st.subheader("Yeo Johnson Charts")
    dfconvert_array = data.to_numpy()
    y = sum(dfconvert_array.tolist(), [])
    fig = plt.figure()
    ax1 = fig.add_subplot(211)

    prob = stats.probplot(y, dist=stats.norm, plot=ax1)
    ax1.set_xlabel('')
    ax1.set_title('Probplot against normal distribution')
    st.pyplot(fig)

    fig = plt.figure()
    ax2 = fig.add_subplot(212)
    xt, lmbda = stats.yeojohnson(y)
    prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    ax2.set_title('Probplot after Yeo-Johnson transformation')
    st.pyplot(fig)
    a = transformation(data)
    standrd = a.standardization()
    standard_data = standrd[1]
    standard_data.insert(loc=0, column='Date', value=date)
    # standard_data.to_csv('latest_data.csv',index=False)

    normalize = a.normalization(data)
    normalize.insert(loc=0, column='Date', value=date)
    st.write("Standardized data (Standard-Scaler)")
    st.write(standard_data)
    st.write("Scaled data (Min-Max)")
    st.write(normalize)
    # normalize.to_csv("latest_data.csv",index=False)

    st.subheader("Choose between following data sets")

    if st.checkbox('Standardized data (Standard-Scaler)'):
        import os
        os.remove("scaler2.pkl")
        standard_data.to_csv('latest_data.csv', index=False)
        joblib.dump("standard", "name.pkl")
    if st.checkbox('Scaled data (Min-Max)'):
        import os
        os.remove("scaler.pkl")
        normalize.to_csv('latest_data.csv', index=False)
        joblib.dump("normal", "name.pkl")
    submit = st.button("Submit")
    if submit:
        st.title("Your final dataset")
        data = pd.read_csv('latest_data.csv')
        data.to_csv("transformed_data.csv")
        st.table(data)


elif page == "Uni-Variate Analysis":
    data = pd.read_csv("latest_data.csv")
    st.header("This is the Final Preprocessed data")
    st.table(data.head())
    st.subheader("Build Models")
    target = joblib.load('target.pkl')
    st.write("This is your selected Target variable:", target)
    data = data[["Date", target]]
    data["Date"] = pd.to_datetime(data["Date"])  # ,format="%d/%m/%Y")
    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime('%Y-%m-%d')
    data = data.sort_values(by="Date")
    data.to_csv("model.csv", index=False)
    models = ["Arima", "Holts-Winter", "FB Prophet"]
    model = st.selectbox("Please select one of the models", models)
    if model == "Arima":
        data = pd.read_csv("model.csv")
        data = data[data['Monthly_sales_total'] > 0.0001]
        data.reset_index(inplace=True)
        data = data.drop("index", axis=1)
        arima_model1 = Arima_Model(data)
        arima_results = arima_model1.arima_model()
        st.subheader("Model Results")
        st.write("Variance:", arima_results[3])
        st.write("Variance Seasonal:", arima_results[4])
        st.write("Variance Trend:", arima_results[5])
        st.write("Variance Residual:", arima_results[6])
        st.write("MAE:", arima_results[8])
        st.write("MSE:", arima_results[9])
        st.write("RMSE:", arima_results[10])
        st.write("MAPE:", arima_results[11])
        st.write("R2:", arima_results[12])
        #st.write("Predictions CSV")
        df = arima_results[13]
        st.table(df.T)

    elif model == "Holts-Winter":
        data = pd.read_csv("model.csv")
        data = data[data['Monthly_sales_total'] > 0.0001]
        data.reset_index(inplace=True)
        data = data.drop("index", axis=1)
        saledata = data["Monthly_sales_total"]
        #dates =data['Date']
        #split = TimeSeriesSplit(n_splits=2)
        train, test = train_test_split(saledata, test_size=0.2, shuffle=False)
        # print(train.isnull().values.any())
        fit = holt_winters(train, crossvalidation=5).fit(use_boxcox=False)
        fit.fittedvalues.plot(style='--', color='red')
        rmse = mean_squared_error(fit.forecast(len(test)), test)
        fit.forecast(len(test)).plot(style='--', color='green', legend=True)
        plt.show()
        st.pyplot()

    elif model == "FB Prophet":
        data = pd.read_csv("model.csv")
        data = data.rename(columns={'Date': 'ds',
                                    'Monthly_sales_total': 'y'})
        train = data.head(150)
        test = data.tail(4)
        model = Prophet(interval_width = 0.95,weekly_seasonality=True,changepoint_prior_scale=0.01,seasonality_mode='multiplicative')
        model.fit(train)
        future_dates = model.make_future_dataframe(periods = 8,freq='M')
        print("First week to forecast.")
        # print(future_dates.tail(5))
        forecast = model.predict(future_dates)
        #print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
        fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})
        st.write("Model Forecast:", fc.tail(10))
        model.plot(forecast)
        st.pyplot()

elif page == "Bi-Variate Analysis":
    st.subheader("Bi-Variate Analysis using Linear Regression")
    data = pd.read_csv("latest_data.csv")

    #target=st.selectbox("Please select target variable",data.columns.tolist())
    target = joblib.load('target.pkl')
    feat1 = st.selectbox("Please select feature 1", data.columns.tolist())
    feat2 = st.selectbox("Please select feature 2", data.columns.tolist())
    if feat1 != "Date" and feat2 != "Date":
        X = data[feat1].values.reshape(-1, 1)
        Y = data[target].values.reshape(-1, 1)
        tips = sns.load_dataset("tips")
        ax = sns.regplot(x=data[feat1], y=data[target], data=tips)
        st.pyplot()
        ax = sns.regplot(x=data[feat2], y=data[target], data=tips)
        st.pyplot()
        st.markdown("Linear Regression")
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.show()
        st.pyplot()

        X = data[feat2].values.reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.show()
        st.pyplot()

elif page == "Multi-Variate Analysis":
    st.markdown("Please select the type of Technique")
    subpage = st.selectbox("Please select the type of model", [
                        'Variable-Importance', 'Linear Regression', 'Ridge Regression and Optimization'])
    st.success("Technique Selected")
    if subpage == "Variable-Importance":
        st.header("Variable-Importance")
        st.subheader("Co-relation plot")
        data = pd.read_csv("latest_data.csv")
        data = data.drop('Date', axis=1)
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
        )
        st.pyplot()
        st.subheader("Feature importance using Boruta trees")
        #target=st.selectbox("Please select target variable",data.columns.tolist())
        target = joblib.load('target.pkl')
        y = data[target]
        X = data.drop(target, axis=1)
        features = list(X.columns)

        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)

        # define Boruta feature selection method
        feat_selector = BorutaPy(
            rf, n_estimators='auto', verbose=2, random_state=1)

        feat_selector.fit(X.values, y.values)

        # check selected features - first 5 features are selected
        # feat_selector.support_

        # check ranking of features
        # print(feat_selector.ranking_)
        X_filtered = feat_selector.transform(X.values)

        final_features = list()
        indexes = np.where(feat_selector.support_ == True)
        for x in np.nditer(indexes):
            final_features.append(features[x])
        st.write("Selected Features through Bortuta trees:", final_features)

        st.subheader("Feature importance using Decision Trees")
        # define the model
        model = DecisionTreeRegressor()
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_

        features2 = X.columns
        score = []
        # summarize feature importance
        for i, v in enumerate(importance):
            score.append(v)
        final = pd.DataFrame(list(zip(features2, score)),
                             columns=['Features', 'Score/Importance'])
        final = final.sort_values(by=['Score/Importance'], ascending=False)
        st.table(final)
        # plot feature importance
        #pyplot.bar([x for x in range(len(importance))], importance)
        # pyplot.show()
        final.plot.bar(x='Features', y='Score/Importance', rot=0)
        plt.xticks(rotation=15)
        st.pyplot()

    elif subpage == "Linear Regression":
        st.subheader("Linear Regression Model")
        data = pd.read_csv("latest_data.csv")
        feat = joblib.load("features.pkl")
        internal = joblib.load("internal.pkl")
        external = joblib.load("external.pkl")
        target = [joblib.load("target.pkl")]
        # joblib.dump(features,"features.pkl")
        if feat != [] and target != []:
            linear_model1 = linear_regression_Model(data)
            linear_results = linear_model1.linear_model(feat, target)
            st.subheader("Linear Regression Results")
            st.write(linear_results[0])
            st.write(linear_results[1])
            st.write(linear_results[2])
            st.write(linear_results[3])
            st.write(linear_results[4])
            st.write(linear_results[5])
            st.write(linear_results[6])
            st.write(linear_results[7])
            st.write(linear_results[8])
            st.write(linear_results[9])

            st.write("Intercept Value")
            st.write(linear_results[11])
            st.table(linear_results[10])

    elif subpage == "Ridge Regression and Optimization":
        #st.subheader("Ridge Regression Model")
        data = pd.read_csv("latest_data.csv")
        data2 = pd.read_csv("Product_data.csv")
        feat = joblib.load("features.pkl")
        internal = joblib.load("internal.pkl")
        external = joblib.load("external.pkl")
        target = [joblib.load("target.pkl")]
        if internal != [] and target != []:
            ridge_model1 = ridge_regression_Model(data)
            ridge_results = ridge_model1.ridge_model(feat, target)
            st.subheader("Ridge Regression Results")
            st.subheader("Train Data Results")
            for i in range(6):
                st.write(ridge_results[i])

            st.subheader("Test Data Results")
            for i in range(6, 12):
                st.write(ridge_results[i])

            ridge_results[12]
            c = [-x for x in coef.values()]
            joblib.dump(ridge_results[12],'ridge_coef.pkl') 
        elif(target != []):
            st.write("Please select atleast one internal variable")
        else:
            st.write("Please select a target variable ")
elif page == 'Scenario Analysis':
    data = pd.read_csv("latest_data.csv")
    data2 = pd.read_csv("Product_data.csv")		
    feat = joblib.load("features.pkl")
    internal = joblib.load("internal.pkl")
    external = joblib.load("external.pkl")
    target = [joblib.load("target.pkl")]
    st.subheader("Scenario Analysis")
    st.write("Feature Range")
    number = len(internal)
    ranges = {}
    for count, ele in enumerate(internal):
        minval = data2[ele].min()
        maxval = data2[ele].max()
        stddev = int(data2[ele].std())

        minval = np.int64(minval)
        maxval = np.int64(maxval)
        try:
            maxval = list(set(maxval))
            maxval = maxval[0]
        except:
            pass

        a = float(minval)
        b = float(maxval)
        d = (a, b)
        a = st.slider(feat[count], float((minval)-2*stddev),
                        float((maxval)+2*stddev), 0.1)
        ranges.update({ele: a})

    features = symbols(' '.join([feature for feature in feat]))
    coef = joblib.load('ridge_coef.pkl')
    expression = coef['intercept']
    for key, value in zip(features, coef.values()):
        print(key, value)
        print(type(key))
        expression = expression + float(value)*key #writing a symboic equation

    st.subheader('The sales equation:')
    small_expression = N(expression,4) #rounding off for displaying in the UI
    st.text('Sales = ' + str(small_expression))
    try:
        scaler = load(open('scaler.pkl', 'rb'))
    except:
        scaler = load(open('scaler2.pkl', 'rb'))
    name = joblib.load('name.pkl')
    if name == "standard":
        scaler = preprocessing.StandardScaler()
    else:
        scaler = preprocessing.MinMaxScaler()
    
    data_internal = data[data.columns.intersection(internal)]
    data_external = data[data.columns.intersection(external)]
    target_col = data2[data2.columns.intersection(target)]
    scaler_internal = scaler.fit(data_internal)
    dictlist=[]
    for key, value in ranges.items():
        dictlist.append(value)
    df=pd.DataFrame(dictlist)
    df=df.T

    inv_val = scaler.transform(df)[0]
    feature_values = [(feature, value) for feature, value in zip(ranges.keys(),inv_val)]
    for feature in external:
        feature_values.append((feature,(random.uniform(0,1))))
    sales = np.array(expression.subs(feature_values))
    sales = sales.reshape(1, -1)
    scaler_target =scaler.fit(target_col)
    sales = scaler_target.inverse_transform(sales)[0][0]
    st.write('sales :',float(sales))
elif page == 'Optimum Variable Value Selection':
    data = pd.read_csv("latest_data.csv")
    data2 = pd.read_csv("Product_data.csv")		
    feat = joblib.load("features.pkl")
    internal = joblib.load("internal.pkl")
    external = joblib.load("external.pkl")
    target = [joblib.load("target.pkl")]
    st.write("Feature Range")
    number = len(feat)
    ranges = {}
    for count, ele in enumerate(internal):
        minval = data2[ele].min()
        maxval = data2[ele].max()
        stddev = int(data2[ele].std())

        minval = np.int64(minval)
        maxval = np.int64(maxval)
        try:
            maxval = list(set(maxval))
            maxval = maxval[0]
        except:
            pass

        a = float(minval)
        b = float(maxval)
        d = (a, b)

        a = st.slider(feat[count], abs(float((minval)-2*stddev)),
                        float((maxval)+2*stddev), d, 0.1)
        ranges.update({ele: a})
    st.write("These are the selected ranges", ranges)
    dictlist = []
    for key, value in ranges.items():
        dictlist.append(value)
    df = pd.DataFrame(dictlist)

    df = df.T
            #df['target'] = 0

    try:
        scaler = load(open('scaler.pkl', 'rb'))
    except:
        scaler = load(open('scaler2.pkl', 'rb'))
    imp_data = pd.read_csv('imputed_data.csv')
    data = imp_data[imp_data.columns.intersection(feat)]
    target_col = imp_data[imp_data.columns.intersection(target)]
    name = joblib.load('name.pkl')
    if name == "standard":
        scaler = preprocessing.StandardScaler()
    else:
        scaler = preprocessing.MinMaxScaler()
    scaler = scaler.fit(data)
    inv_val = scaler.transform(df)
    inv_val = pd.DataFrame(inv_val)

    dictlist = list(inv_val.apply(tuple, axis=0))
        # dictlist=dictlist[:-1]

    listofzeroes = [0] * len(feat)
    Price_Per_Unit_index = feat.index('Price_Per_Unit')
    listofzeroes[Price_Per_Unit_index] = 1
    Discount_index = feat.index('Discount')
    listofzeroes[Discount_index] = -1

    A = listofzeroes
    A = [-x for x in A]
    A = [A]
    b = 0.85

    res = linprog(c, A_ub=A, b_ub=b, bounds=dictlist)

    output = res.x
    res_list = [output[i] * c[i] for i in range(len(output))]
    st.write(res_list)
    sales = sum(res_list)

    if name == "standard":
        scaler = preprocessing.StandardScaler()
    else:
        scaler = preprocessing.MinMaxScaler()
    scaler = scaler.fit(target_col)
    inv_tar = scaler.transform(target_col)
    sales_transformed = scaler.inverse_transform(sales.reshape(-1, 1))
    sales_transformed = sales_transformed[0]
    sales_transformed = sales_transformed[0]
    sales_transformed = -sales_transformed
    st.header("Final Sales Prediction")
    st.write("MAX SALES:", sales_transformed)

