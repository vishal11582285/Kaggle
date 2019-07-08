
# coding: utf-8

# In[1]:


# %load ../titanic/mlpack.py
#Suppress Warninings
import warnings
import requests 
import json
from dask import delayed, compute
import dask
from xml.etree import ElementTree
from matplotlib import pyplot as plt
# %matplotlib notebook

from scipy.stats import zscore
import collections

from dask.diagnostics import ProgressBar
ProgressBar().register()

warnings.filterwarnings('ignore')

#Simpled desiged function to give a birds eye look into the dataset.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sb

from sklearn import model_selection
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,r2_score,classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold,StratifiedKFold, cross_val_score
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

def read_basic(name=None):
    if(name==None):
        raise ValueError('No File Name provided')
        return 0
    print('-------------------{}----------------'.format(name))
    df = pd.read_csv(name)
    print('Datatypes: \n', df.dtypes)
    print('-----------------------------\n')
    print('Dimension: \n', df.shape)
    print('-----------------------------\n')
    print('Missing values per column: \n', df.isna().sum())
    print('-----------------------------\n')
    print('Dataframe Info: \n', df.info())
    print('-----------------------------\n')
    print('Descriptive Stats: \n', df.describe())

    return df# %load mlpack.py


# AS per the second dataset on Kaggle, the block code information can be obtained using the API:
# https://geo.fcc.gov/api/census/#!/block/get_block_find. We use this to find the appropraiate block code
# for all lat,long pairs in the NYC dataset.
# @delayed
# def fips_code(lat, long):
#     url = 'https://geo.fcc.gov/api/census/block/find?latitude={}&longitude={}&format=json'.format(lat,long)
#     r = str(requests.get(url = url).content)
# #     print(r)
# #     print(r.find('FIPS'))
#     return np.int(r[r.find('FIPS')+7:r.find("bbox")-3])
# #     msg = r.content.decode()
# #     return np.int(json.loads(msg)['Block']['FIPS'])


#THis task took around 2 hours, so I have saved the result to re-use in a .npy format
def fips_code(lng, lat): 
    '''Returns census block code given longitude and latitude.'''
    params = {'longitude': str(lng), 'latitude': str(lat)}
    r = requests.get("http://data.fcc.gov/api/block/2010/find", params=params)
    tree = ElementTree.fromstring(r.content)

    for child in tree.iter('{http://data.fcc.gov/api}Block'):
        return child.attrib['FIPS']


# In[2]:


#Basic read csv file to data frames

nyc_foursquare_df = read_basic('dataset_TSMC2014_NYC.csv')
nyc_foursquare_df['CensusTract'] = np.load('block_code_nyc.npy')
nyc_foursquare_df.CensusTract = nyc_foursquare_df.CensusTract.apply(lambda x: str(x)[:11])
nyc_foursquare_df.head()


# ## Read and process source data into pandas dataframe

# In[3]:


#1.	Import NYC check-in dataset(dataset_TSMC2014_NYC.csv) into python
nyc_census_checkin_df = nyc_foursquare_df[['latitude', 'longitude', 'CensusTract']]

# 2.	Convert Coordinates in the dataset to census tracts and output the file to census_tracts_per_checkin.csv
# 3.	Add a census tract column to the census_tracts_per_checkin.csv file. Make sure this column is the same as the dataset_TSMC2014_NYC.csv file
nyc_census_checkin_df.to_csv('census_tracts_per_checkin.csv')


# In[4]:


# 4.	Import NYC census dataset into python(nyc_census_tracts.csv)
nyc_census_tracts_df = read_basic('nyc_census_tracts.csv')


# In[5]:


# 5.	Create a dataset from 4 that excludes all null values of the income variable from nyc_census_tracts.csv and 
nyc_census_tracts_df = nyc_census_tracts_df[~nyc_census_tracts_df.Income.isna()]
print('Shape of Census Tract dataset: ',  nyc_census_tracts_df.shape)
nyc_census_tracts_df.isna().sum()


# In[6]:


# 6.	Create a separate dataset consisting of the census_tract column alone
census_tract_only_df = nyc_census_checkin_df[['CensusTract']]
census_tract_only_df.info()

# 7.	Calculate the total number of checkins per census tract and include it as a separate column in 6
check_in_counts = collections.Counter(census_tract_only_df.CensusTract.values)
census_tract_only_df['num_checkins'] = census_tract_only_df.CensusTract.apply(lambda x: check_in_counts[x])
census_tract_only_df.CensusTract = census_tract_only_df.CensusTract.apply(np.int)
census_tract_only_df.info()
census_tract_only_df.head()

#It is a good idea to drop duplicates
# print()
census_tract_only_df = census_tract_only_df.drop_duplicates()
census_tract_only_df.info()
census_tract_only_df.head()
# census_tract_only_df.num_checkins.isna().sum()


# In[7]:


# 8.	Merge datasets created in 5 and 7
print('SHapes before merge: ', nyc_census_tracts_df.shape, census_tract_only_df.shape )
nyc_merge_census_tract_df = nyc_census_tracts_df.merge(census_tract_only_df, on='CensusTract', how='inner')
print('Shape of merge dataset: ',nyc_merge_census_tract_df.shape)
nyc_merge_census_tract_df.head()


# In[8]:


nyc_merge_census_tract_df.isna().sum()


# In[9]:


# 9.	Remove census_tract, county and borough columns from the merged dataset in 8
nyc_merge_census_tract_df.drop(['CensusTract','Borough','County'], axis=1, inplace=True)


# In[10]:


nyc_merge_census_tract_df = nyc_merge_census_tract_df.reindex()
# nyc_merge_census_tract_df.info()


# In[11]:


nyc_merge_census_tract_df['Z_num_checkins'] = zscore(nyc_merge_census_tract_df.num_checkins.values)


# In[12]:


# 10. Eliminate outliers from the num_checkins column 
#Lets consider those that lie 3 standard deviations away from mean as outliers.
nyc_merge_census_tract_df = nyc_merge_census_tract_df[abs(nyc_merge_census_tract_df.Z_num_checkins)<3]
# nyc_merge_census_tract_df.info()
nyc_merge_census_tract_df.isna().sum()
nyc_merge_census_tract_df.info()


# In[13]:


nyc_merge_census_tract_df.MeanCommute.plot(kind='hist', title='Histogram of Mean Commute')


# In[14]:


# 11.	Impute or remove missing values as you deem fit

#We observe 1 missing value for ChildPoverty and 3 missing values for MeanCOmmute
#Impute ChildPoverty with 0, as 0 percent is the mode
#Impute expcted value=mean of MeanCommute field as its roughly follows a normal distribution
nyc_merge_census_tract_df.ChildPoverty.fillna(0, inplace=True)
nyc_merge_census_tract_df.drop(['Z_num_checkins'], axis=1, inplace=True)
nyc_merge_census_tract_df.MeanCommute.fillna(nyc_merge_census_tract_df.MeanCommute.mean(), inplace=True)


# In[15]:



nyc_merge_census_tract_df.info()


# In[16]:


# 13.	Split the dataset into dependent and independent variables, with the dependent variable being total number of check-ins

nyc_census_X = nyc_merge_census_tract_df.drop(['num_checkins'],axis=1)
nyc_census_Y = nyc_merge_census_tract_df['num_checkins']


# In[17]:


nyc_census_Y.shape


# In[18]:


# 14.	Do min-max scaling on the independent variables
from sklearn.preprocessing import MinMaxScaler

def scale_min_max(series):
    scaler_ = MinMaxScaler()
    return scaler_.fit_transform(series.values.reshape(-1,1))


# In[19]:


import seaborn as sb
for col in nyc_census_X.columns:
    nyc_census_X[col] = scale_min_max(nyc_census_X[col])
# print(min(nyc_census_Y), max(nyc_census_Y))
nyc_census_X.head()


# In[20]:


# nyc_census_Y.reshape(1,-1)[0]


# In[21]:


plt.figure(figsize=(20,15))
# Generate a mask for the upper triangle

# correlation heatmap of dataset
def correlation_heatmap(df):
    return df.corr()

# 12.	Construct a correlogram of all the variables, output the correlation values to an external excel file
nyc_merge_census_corr = correlation_heatmap(nyc_merge_census_tract_df)
nyc_merge_census_corr.to_csv('nyc_pearson_correlation.csv')
nyc_merge_census_corr.columns

mask = np.zeros_like(nyc_merge_census_corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(nyc_merge_census_corr.corr(), mask=mask, vmin=-1, vmax=1, center=0, cmap=sb.diverging_palette(20, 220, n=200),square=True)


# In[22]:


# plt.figure(figsize=(20,18))
# sb.heatmap(nyc_census_X.corr())


# In[23]:


# nyc_census_X = nyc_census_X[['MeanCommute', 'Income', 'Walk', 'PublicWork', 'Drive',
#        'Carpool']]


# In[24]:


# 15.	Fit an extra-trees model of the data and display the relative importance of the features
MLA = [
    #Ensemble Methods
    ExtraTreesRegressor()]


# In[25]:


nyc_census_X_train, nyc_census_X_test, nyc_census_Y_train, nyc_census_Y_test =  train_test_split(nyc_census_X, nyc_census_Y, test_size=0.3, random_state=0)


# In[26]:


# nyc_census_Y_test.isna().sum()

def mean_squared_error(y, y_pred):
# assuming y and y_pred are numpy arrays
#     print(type(y), type(y_pred))
    return np.mean(np.sqrt(np.sum(np.power(y - y_pred, 2))))


# In[27]:


#create table to compare MLA metrics
def perform_algorithm_analysis(MLA):
    from sklearn.metrics import make_scorer
    my_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    
    MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean','MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #create table to compare MLA predictions
    MLA_predict = pd.DataFrame()
    
    #5-fold cross validation
    cv_split = model_selection.ShuffleSplit(n_splits = 5, random_state = 0 ) 
    
    #index through MLA and save performance to table
    row_index = 0
    for alg in MLA:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg, nyc_census_X_train, nyc_census_Y_train, cv  = cv_split, scoring='explained_variance', return_train_score=True)
    #     print(cv_results)
        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
        #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

        #save MLA predictions - see section 6 for usage
        alg.fit(nyc_census_X_train, nyc_census_Y_train)
        row_index+=1
        MLA_predict[MLA_name] = alg.predict(nyc_census_X_test)

    #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    return MLA, MLA_compare, MLA_predict


# In[28]:


MLA, MLA_compare, MLA_predict = perform_algorithm_analysis(MLA)

model = MLA[0]
scores_ = model.feature_importances_
feat = nyc_census_X.columns

list_ = sorted([(y,x) for x,y in zip(feat, scores_)], reverse=True)
top_importance = [y for x,y in list_]
print(list_)
# plt.bar(feat, scores_)


# In[29]:


plt.figure(figsize=(12,10))
plt.title('Feature Importance after Extra Trees CLassifier')
plt.barh([x[1] for x in list_], [x[0] for x in list_])
# plt.show()


# In[30]:


#Machine Learning Algorithm (MLA) Selection and Initialization
# 20.	Fit SVR, linear SVR, gradient boosting, ada-boost, SGD-regressor, lasso and elastic net models 
# on the dataset and measure the cross-validation accuracy scores

from sklearn import ensemble, linear_model, naive_bayes, svm, tree, discriminant_analysis, gaussian_process, neighbors
from sklearn import model_selection
from xgboost import XGBClassifier, XGBRegressor

MLA = [
    #Ensemble Methods
    ensemble.AdaBoostRegressor(n_estimators=100, loss='exponential'),
    ensemble.ExtraTreesRegressor(),
    ensemble.GradientBoostingRegressor(n_estimators=100),
    ensemble.RandomForestRegressor(),

    #GLM
    linear_model.LinearRegression(),
    linear_model.SGDRegressor(),
    linear_model.Ridge(),
    linear_model.Lasso(),
    
    #SVM
    svm.SVR(),
    svm.LinearSVR(),
    
    #
    ]


# In[31]:


np.array(top_importance[0:15])


# In[32]:


print(top_importance[0:15])
nyc_census_X[top_importance[0:15]].corr()


# In[33]:


#We remove the most correlated feature, where correlation is more than +/- 0.8
# 16.	Eliminate unimportant features
top_importance = ['MeanCommute', 'Walk', 'Income', 'Service', 'Drive', 'PrivateWork', 'Construction', 'OtherTransp', 'Employed']


# In[34]:


#Looking at the correlations, we are good with the selected featured for our prediction
nyc_census_X[top_importance].corr()


# In[35]:


nyc_census_X = nyc_census_X[top_importance]


# In[36]:


max_vals = list()
# for x in range(1,len(top_importance)):
selected_features = top_importance[0:]
# print(selected_features)
nyc_census_X_ = nyc_census_X[selected_features]
nyc_census_X_train, nyc_census_X_test, nyc_census_Y_train, nyc_census_Y_test =  train_test_split(nyc_census_X_, nyc_census_Y, test_size=0.3, random_state=0)
MLA, MLA_compare, MLA_predict = perform_algorithm_analysis(MLA)
# max_vals.append(MLA_compare['MLA Test Explained Variance'].max())


# In[37]:


MLA_compare


# In[38]:


import statsmodels.api as sm
# from pandas.stats.api import ols


# In[39]:


#Fitting Ordinary least squares
# 17.	Fit a Regression model and further eliminate any unnecessary features
model_ols = sm.OLS(nyc_census_Y_train, nyc_census_X_train)
# model = sm.OLS(y, X)
results = model_ols.fit()
print(results.summary())


# In[40]:


#Removig the predictors since the p-value>0.05
nyc_census_X.drop(['OtherTransp'], axis=1, inplace=True)
nyc_census_X.drop(['Service'], axis=1, inplace=True)
# nyc_census_X.drop(['IncomePerCap'], axis=1, inplace=True)


# In[41]:


# 18.	Split the dataset into training and testing
nyc_census_X_train, nyc_census_X_test, nyc_census_Y_train, nyc_census_Y_test =  train_test_split(nyc_census_X, nyc_census_Y, test_size=0.3, random_state=0)


# In[42]:


# 19.	Do cross-validation
MLA, MLA_compare, MLA_predict = perform_algorithm_analysis(MLA)
MLA_compare


# In[43]:


model_ols = sm.OLS(nyc_census_Y_train, nyc_census_X_train)
# model = sm.OLS(y, X)
results = model_ols.fit()
print(results.summary())


# In[44]:


# 21.	Fit the model with the best cross-validation accuracy to the test data.
# 22.	Output the prediction results to an excel sheet.
model = MLA[2] #Best performing model for us. GradientBoostingRegressor
submission = pd.DataFrame({'Actual CheckIns': nyc_census_Y_test, 'Predicted CheckIns': model.predict(nyc_census_X_test)})
submission.to_csv('submissionNYCRang.csv')


print('---------Done----------------')

