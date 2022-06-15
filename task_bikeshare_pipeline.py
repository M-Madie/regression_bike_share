import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

# load data, delete unused columns
df = pd.read_csv('./train.csv', index_col=0, parse_dates=True)
del df['atemp']
df.loc[df['weather']==4] = 3
df.drop(labels='2012-01-09 18:00:00', axis=0, inplace=True)

# split to X, and y
X = df.iloc[:,0:7]
y = df.iloc[:,7:9]


# Add new columns to X
X['time'] = df.index.hour
X['month'] = df.index.month_name()
X['year'] = df.index.year


# Log labels (y) due to heterocedasticity
y['registered'] = np.log(y['registered'], where=(y['registered']!=0))
y['casual'] = np.log(y['casual'], where=(y['casual']!=0))


### Model 1 - Predicting Casual Users
# feature engineering for numerical
numeric_features1 = ['temp', 'humidity', 'windspeed']
numeric_transformer1 = make_pipeline(
    MinMaxScaler()
    )

# feature engineering for categorical
categorical_features1 = ['year', 'time' ,'month', 'season', 'weather', 'workingday', 'holiday']
categorical_transformer1 = OneHotEncoder(handle_unknown="ignore")

# polynomial transformations
pol_features1 = ['time']
pol_transformer1 = make_pipeline(
    MinMaxScaler(),
    PolynomialFeatures(degree= 2, include_bias= False,interaction_only=False)
    )

# preprocessor
preprocessor1 = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer1, numeric_features1),
        ("poly", pol_transformer1, pol_features1),
        ("cat", categorical_transformer1, categorical_features1),
    ],
    remainder='passthrough')

# create the model pipeline
pipeline1 = make_pipeline(preprocessor1, LinearRegression())



### Model 2 - Predicting Registered Users
# feature engineering for numerical
numeric_features2 = ['temp', 'humidity', 'windspeed']
numeric_transformer2 = make_pipeline(
    MinMaxScaler(),
    PolynomialFeatures(degree= 3, include_bias= False,interaction_only=False)
    )

# polynomial transformations
pol_features2 = ['time']
pol_transformer2 = make_pipeline(
    MinMaxScaler(),
    PolynomialFeatures(degree= 4, include_bias= False,interaction_only=False)
    )

# feature engineering for categorical
categorical_features2 = ['year', 'time', 'month', 'season', 'workingday', 'holiday']
categorical_transformer2 = OneHotEncoder(handle_unknown="ignore")

# preprocessor
preprocessor2 = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer2, numeric_features2),
        ("poly", pol_transformer2, pol_features2),
        ("cat", categorical_transformer2, categorical_features2),
    ],
    remainder='passthrough')

# create the model pipeline
pipeline2 = make_pipeline(preprocessor2, Ridge())



# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 20)
ycas_train = y_train['casual']
ycas_test = y_test['casual']
yreg_train = y_train['registered']
yreg_test = y_test['registered']



#### Fit model1 - Casual 
# fit the pipeline to training data    (X_train, ycas_train)
pipeline1.fit(X_train, ycas_train)
print("model1(casual users) score(Train): %.3f" % pipeline1.score(X_train, ycas_train))

###
# calculate the accuracy score from test data (X_test, ycas_test)
print("model1 score(Test-Validation): %.3f" % pipeline1.score(X_test, ycas_test))


#### Fit model2 - Registered 
# fit the pipeline to training data    (X_train, yreg_train)
pipeline2.fit(X_train, yreg_train)
print("model2(registered users) score(Train): %.3f" % pipeline2.score(X_train, yreg_train))

###
# calculate the accuracy score from test data (X_test, yreg_test)
print("model2 score(Test-Validation): %.3f" % pipeline2.score(X_test, yreg_test))



# get predictions from the pipeline
#print(pipeline1.predict(X_test))


