
# coding: utf-8

# In[1]:


## Task:

# Look at https://github.com/sundeepblue/movie_rating_prediction and download data file movie_metadata.csv.

# Then

 

#     Get additional data from other sources if required.
#     Perform Data Preprocessing and Exploratory Data Analysis which includes data visualization also.
#     Create at least 3 different machine learning models to predict IMDB rating of a movie.
#     Compare the results and suggest the model which could be useful to deploy into production.
#     Optional: you can also use TensorFlow, Keras or Pytorch to build the models.  

 

# Send your work (code and results and process documentation) as Python notebook output (.html or .ipynb files).


# In[2]:


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
plt.rcParams['figure.figsize']=(12,8)

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
    print('Missing values per column (Percentage): \n', (df.isna().sum()/df.shape[0])*100)
    print('-----------------------------\n')
    print('Dataframe Info: \n', df.info())
    print('-----------------------------\n')
    print('Descriptive Stats: \n', df.describe())

    return df# %load mlpack.py


# In[3]:


imdb_db = read_basic('movie_metadata.csv')


# In[4]:


imdb_db.head()


# In[5]:


#Lets check if the data types are in order 
print(imdb_db.dtypes)
# imdb_db.columns


# In[6]:


# #Documents that do not have title_year are probably mis-reported and currently there are 2 percent of the data with
#such problems, so we can safely drop them
missing = imdb_db.title_year.dropna().index.tolist()
imdb_db = imdb_db.loc[missing,:]


# In[7]:


imdb_db.reset_index()
imdb_db.info()
#MIssing values (Percentage wise):

# Missing values per column (Percentage): 
# color                         0.376760
# director_name                 2.062265
# num_critic_for_reviews        0.991473
# duration                      0.297442
# director_facebook_likes       2.062265
# actor_3_facebook_likes        0.456078
# actor_2_name                  0.257783
# actor_1_facebook_likes        0.138806
# gross                        17.529248
# genres                        0.000000
# actor_1_name                  0.138806
# movie_title                   0.000000
# num_voted_users               0.000000
# cast_total_facebook_likes     0.000000
# actor_3_name                  0.456078
# facenumber_in_poster          0.257783
# plot_keywords                 3.033908
# movie_imdb_link               0.000000
# num_user_for_reviews          0.416419
# language                      0.237954
# country                       0.099147
# content_rating                6.008328
# budget                        9.756098
# title_year                    2.141582
# actor_2_facebook_likes        0.257783
# imdb_score                    0.000000
# aspect_ratio                  6.523895
# movie_facebook_likes          0.000000


# In[8]:


imdb_db.color = imdb_db.color.astype('category')
imdb_db.title_year = imdb_db.title_year.astype(np.int)


# In[9]:


# The most number of missing values are in gross column: about ~ 17 percent, which we cannot straight away drop, 
# as we risk losing lot of data. Gross is a float object. Let us see how it fares.

imdb_db.gross.agg([np.mean, np.median, np.min, np.max])


# In[10]:


print(imdb_db.corr()['gross'])
#We see that gross has a maximum correlation with num_voted_users of about 0.64/
#Lets create a simple Linear regression model between (num_voted_users, gross) to impute missing values
imdb_num_users_gross = imdb_db[~imdb_db.num_voted_users.isna()]
imdb_num_users_gross = imdb_db[~imdb_db.gross.isna()]
# imdb_num_users_gross.gross.isna().sum()


# In[11]:


model = LinearRegression()
model.fit(imdb_num_users_gross[['num_voted_users']], imdb_num_users_gross[['gross']])
# imdb_db.plot(x='num_voted_users', y='gross', kind='scatter')


# In[12]:


from sklearn.metrics import r2_score
r2_score(imdb_num_users_gross[['gross']], model.predict(imdb_num_users_gross[['num_voted_users']]))
#This model has an acceptable R2 score. We use it to impute gross value.


# In[13]:


gross_vals = pd.Series(model.predict(imdb_db[['num_voted_users']]).ravel())
imdb_db.gross = imdb_db.gross.combine_first(gross_vals)


# In[14]:


print('Missing values per column (Percentage): \n', (imdb_db.isna().sum()/imdb_db.shape[0])*100)


# In[15]:


#For budget, lets impute the missing values with the median
imdb_db.budget = imdb_db.budget.fillna(imdb_db.budget.median())

#For aspect_ratio, lets impute the missing values with the mode, as value_counts show 
#the distribution is maximum for mode value = 2.35

imdb_db.aspect_ratio = imdb_db.aspect_ratio.fillna(imdb_db.aspect_ratio.mode())


# In[16]:


print('Missing values per column (Percentage): \n', (imdb_db.isna().sum()/imdb_db.shape[0])*100)
#LLets target content_rating which has 5 percent missing values


# In[17]:


#We have below cross tab data based on genres vs content_rating
crosstab_genres_rating = pd.crosstab(imdb_db.genres.apply(lambda x: x.split('|')[0]), imdb_db.content_rating)
crosstab_genres_rating


# In[18]:


imdb_db.content_rating.value_counts().plot(kind='bar')


# In[19]:


print(crosstab_genres_rating.index, crosstab_genres_rating.columns)
# 'Approved', 'G', 'GP', 'M', 'NC-17', 'Not Rated', 'PG', 'PG-13',
#        'Passed', 'R', 'TV-14', 'TV-G', 'TV-PG', 'Unrated', 'X'
dict_ = {'Action': 'PG-13', 'Adventure': 'PG', 'Animation': 'PG', 'Biography': 'R', 'Comedy': 'R', 'Crime': 'R',
       'Documentary': 'PG', 'Drama':'R', 'Family': 'PG', 'Fantasy':'R', 'Film-Noir':'Unrated', 'Horror':'R',
       'Music':'R', 'Musical':'PG', 'Mystery':'R', 'Romance':'PG-13', 'Sci-Fi':'R', 'Thriller':'R',
       'Western': 'R'}
new_content_rating = pd.Series(imdb_db.genres.apply(lambda x: dict_[x.split('|')[0]] if x.split('|')[0] in dict_.keys() else 'Unrated'))
imdb_db.content_rating = imdb_db.content_rating.combine_first(new_content_rating)
imdb_db.content_rating.value_counts().plot(kind='bar')


# In[20]:


print('Missing values per column (Percentage): \n', (imdb_db.isna().sum()/imdb_db.shape[0])*100)


# In[21]:


#Now majority of the missing values are imputed, lets proceed with clearing the dataframe from NA values
imdb_db.dropna(inplace=True)
# imdb_db.drop('index', axis=1, inplace=True)
print(imdb_db.shape) #Original Dimension: (5043, 28) #New Dimesion (4729, 28)

#We have dropped (314) rows which is about 6 percent of the original data, which is acceptable.


# In[22]:


#We are without any missing values now and ready to proceed with EDA
print('Missing values per column (Percentage): \n', (imdb_db.isna().sum()/imdb_db.shape[0])*100)


# In[23]:


import seaborn as sb
sb.pairplot(imdb_db)


# In[24]:


plt.figure(figsize=(12,8))
# plt.colormaps
sb.heatmap(imdb_db.corr(),annot=True,cmap = 'PuOr')


# In[25]:


# imdb_db[['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','cast_total_facebook_likes']]

filtered = imdb_db[['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes']]
imdb_db['calc_likes'] = filtered.apply(np.sum,axis=1)
'''
We create a new feature calc_likes which is calculated total of all three actor likes in the database.
As observed in the heatmap below, actor_1_facebook_likes greatly correlates with calc_likes and also the already
provided feature cast_total_facebook_likes.
Therefore, we consider actor_1_facebook_likes to be enough estimator for the target withoutt the need for other 
features: actor_2_facebook_likes, actor_3_facebook_likes, cast_total_facebook_likes. Lets proceed to dropping them.
'''

imdb_db.drop(['actor_2_facebook_likes','actor_3_facebook_likes','cast_total_facebook_likes', 'calc_likes'], axis=1, inplace=True)


# In[26]:


sb.heatmap(imdb_db.corr(),annot=True,cmap = 'PuOr')


# In[27]:


#Lets drop num_voted_users from analysis since it is correlated to another continuous feature num_user_for_reviews


# In[28]:


imdb_db.drop(['num_voted_users'], axis=1, inplace=True)


# In[29]:


group_content_rating = imdb_db.groupby('content_rating')['imdb_score'].agg([np.mean, np.sum, np.median])
# group_content_rating.columns
group_content_rating.sort_values(by=['sum','median'], ascending=False)


# In[30]:


#Rating should not be sensitive to the nearest decimal, infact there will be very few movies with a 0

plt.title('IMDB Raitings')
plt.xlabel('Rating (0-10)')
plt.ylabel('PDF(IMDB_Rating)')
sb.distplot(imdb_db.imdb_score, bins=10)

# imdb_db.imdb_score[:10]

# imdb_db.plot('imdb_score', kind='hist')

#Based on the below histogram, it would be wiser to consider this as a classification problems instead of a regression problem.
#We can create 4 classes:
# '''
# Class 0: IMDB Rating 0-4
# Class 1: IMDB Rating 3-5
# Class 2: IMDB Rating 5-8
# Class 3: IMDB Rating 8-10

# '''
#We can create a new predictor to map imdb_rating in the dataset to these customized classes


# In[31]:


#Good R2 score
# def convert_to_class(rating):
#     if(rating<=3):
#         return '1.5'
#     if(rating>3 and rating<=4):
#         return '3.5'
#     if(rating>4 and rating<=5):
#         return '4.5'
#     if(rating>5 and rating<=6):
#         return '5.5'
#     if(rating>6 and rating<=7):
#         return '6.5'
#     if(rating>7 and rating<=8):
#         return '7.5'
#     if(rating>8 and rating<=9):
#         return '8.5'
#     if(rating>9):
#         return '9.5'
#     return '-99'


# In[32]:


def convert_to_class(rating):
    if(rating<=3):
        return '1.5'
    if(rating>3 and rating<=4):
        return '3.5'
    if(rating>4 and rating<=5):
        return '4.5'
    if(rating>5 and rating<=6):
        return '5.5'
    if(rating>6 and rating<=7):
        return '6.5'
    if(rating>7 and rating<=8):
        return '7.5'
    if(rating>8 and rating<=9):
        return '8.5'
    if(rating>9):
        return '9.5'
    return '-99'


# In[33]:


imdb_db['class_rating'] = imdb_db.imdb_score.apply(convert_to_class)
imdb_scores = imdb_db['imdb_score']
# imdb_db.drop('imdb_score', axis=1, inplace=True)


# In[34]:


plt.title('IMDB Class Ratings')
plt.xlabel('Rating Clas')
plt.ylabel('PDF(IMDB_Class Rating)')
plt_params_x = imdb_db.class_rating.value_counts().index
plt_params_y = imdb_db.class_rating.value_counts().values
sb.barplot(plt_params_x, plt_params_y)
imdb_db.class_rating.value_counts()


# In[35]:


# sb.barplot(imdb_db.class_rating.values, hue='color', data=imdb_db)


# In[36]:


import researchpy as rp
from scipy import stats
table, results, expected = rp.crosstab(imdb_db['color'], imdb_db['class_rating'], test= 'chi-square', expected_freqs=True)  
table , results
#Just to verify with Chi-square test of independance if the categorical values 'color' and 'black and white' are independance.
#p-value of <=0.05 indicates, there is indeed independance. So we keep the attribute 'color' and move ahead


# In[37]:


imdb_db.columns


# In[38]:


# import researchpy as rp
# from scipy import stats
# table, results, expected = rp.crosstab(imdb_db['aspect_ratio'], imdb_db['class_rating'], test= 'chi-square', expected_freqs=True)  
# table , results


# In[39]:


#Nomical festures can be dropped like movie_title, #Similarly movie imdb link
imdb_db.drop('movie_title', axis=1, inplace=True)
imdb_db.drop('movie_imdb_link', axis=1, inplace=True)


# In[40]:


imdb_db.aspect_ratio = imdb_db.aspect_ratio.astype('category')


# In[41]:


imdb_db.aspect_ratio.value_counts()
# If we look at the theory, then indeed 2.35 and 1.85 are most normal aspect ratios.
# VIDEO STANDARDS
# Eventually cinema converged on two leading standards: a normal 1.85:1 widescreen and an anamorphic 2.39:1 widescreen. With television, the formats became 4:3 with standard definition and later 16:9 with high definition, which at 1.78:1 
# was a close match to 1.85:1 widescreen cinema. 

#Based on this anything which is not 2.35 or 1.85, can be termed as rare, 
#and then we can enforce one-hot encoding convertion on aspect_ratio

imdb_db.aspect_ratio = imdb_db.aspect_ratio.apply(lambda x: 'rare' if not x in [2.35,1.85] else x)
imdb_db.aspect_ratio.value_counts()


# In[42]:


from sklearn.preprocessing import OneHotEncoder
ohot = OneHotEncoder()
# ohot.fit_transform(imdb_db.aspect_ratio.values)

aspect_dummies = pd.get_dummies(imdb_db.aspect_ratio, prefix='aspect', drop_first=False)
imdb_db = pd.concat([imdb_db, aspect_dummies], axis=1)
imdb_db.drop('aspect_ratio', axis=1, inplace=True)


# In[43]:


imdb_db.info()


# In[44]:


#Feature engineering: INstead of using gross earnings and profits, we will use the ratio:
#(gross/budget) to denote if the movie made more than the budget. In this case, its a hit in business terms.
#Otherwise, its not so much business value
imdb_db['make_profit'] = [1 if x>=2.7 else 0 for x in imdb_db['gross']/imdb_db['budget']]

#Value 2.7 obtained after trying out multiple rations and choosing the best based on contingency crosstab tables and
#corrrelation with class_rating

imdb_db[['make_profit','class_rating']].corr()

#We are now hence free to drop budget and gross fields
imdb_db.drop(['budget', 'gross'], axis=1, inplace=True)


# In[45]:


#WE need to now split genres iinto separate categories and create a binary encoding for
#if the movie falls in one or many genres.

genres_set=set()
for x in imdb_db.genres:
    for y in x.split('|'):
        genres_set.add(y)
genres_set = list(sorted(genres_set))
pd_genres = pd.DataFrame(columns=genres_set)

imdb_db = pd.concat([imdb_db,pd_genres],axis=1)

for genre in genres_set:
    imdb_db[genre] = imdb_db.genres.apply(lambda x: 1 if genre in x else 0)


# In[46]:


# We can confirm that all the genres entries translated correctly to one of the binary
#genre encoding labels
imdb_db[genres_set].apply(lambda x: sum(x), axis=1).value_counts()

#We can drop genres column now
imdb_db.drop(['genres'], axis=1, inplace=True)


# In[47]:


#Lets drop any nominal features, like actor and director names, since their populaarity in anyways captured in
#facebook likes fore respective titles, eg. director_fb_likes,etc.

imdb_db.drop(['actor_1_name', 'actor_2_name', 'actor_3_name','director_name'], axis=1, inplace=True)


# In[48]:


#Lets analyze plot keywords. Tjis would be something that we can perform NLP on.
#For our base machine learning, let us skip this information and keep it for later.
imdb_db.drop(['plot_keywords'], axis=1, inplace=True)
# imdb_db.plot_keywords


# In[49]:


# imdb_db = imdb_db.reset_index()
imdb_db.info()


# In[50]:


#Lets analyze content_rating
#Again just like aspect_ratio, we can turn these categorical variable into one-hot encoding, as
#'R', 'PG-13', and 'PG' are most common content ratings, and rest all can be encoded as 'rare'
imdb_db.content_rating = imdb_db.content_rating.apply(lambda x: 'rare' if not x in ['R','PG-13','PG'] else x)

cr_dummies = pd.get_dummies(imdb_db.content_rating, prefix='cr', drop_first=False)
imdb_db = pd.concat([imdb_db, cr_dummies], axis=1)
imdb_db.drop('content_rating', axis=1, inplace=True)


# In[51]:


#Similar concept for Country
imdb_db.country = imdb_db.country.apply(lambda x: 'rare' if not x in ['USA','UK','France','Canada','Germany'] else x)

cntry_dummies = pd.get_dummies(imdb_db.country, prefix='cntry', drop_first=False)
imdb_db = pd.concat([imdb_db, cntry_dummies], axis=1)
imdb_db.drop('country', axis=1, inplace=True)


# In[52]:


#Similar concept for Language
freq_lang = imdb_db.language.value_counts()[:6].index.values.tolist()
imdb_db.language = imdb_db.language.apply(lambda x: 'rare' if not x in freq_lang else x)

lang_dummies = pd.get_dummies(imdb_db.language, prefix='lang', drop_first=False)
imdb_db = pd.concat([imdb_db, lang_dummies], axis=1)
imdb_db.drop('language', axis=1, inplace=True)


# In[53]:


imdb_db.color.unique()
color_ = {'Color' : 1, ' Black and White':0}

imdb_db.color = imdb_db.color.replace(color_)


# In[54]:


#By the below plot, we see that it makes sense to split the title_year into chunks of 10 years from 1927 to 2016
imdb_db['YearBin'] = pd.cut(imdb_db.title_year,bins=9) 
year_approx = imdb_db.groupby('YearBin')['make_profit'].sum()
year_approx.plot(kind='bar')

imdb_db.YearBin = LabelEncoder().fit_transform(imdb_db.YearBin)

imdb_db.drop('title_year', axis=1, inplace=True)


# In[55]:


#All numerical values can be applied a Standard Scaler, for better and faster convex convergence
from sklearn.preprocessing import StandardScaler
scaler_ = StandardScaler()
for col in ['num_critic_for_reviews', 'duration','director_facebook_likes', 'actor_1_facebook_likes',
       'num_user_for_reviews', 'movie_facebook_likes', 'facenumber_in_poster']:
    imdb_db[col] = scaler_.fit_transform(imdb_db[col].values.reshape(-1,1))
imdb_db.describe()

imdb_db.to_csv('processed_imdb.csv')


# In[56]:


plt.figure(figsize=(12,8), dpi=600)
sb.heatmap(imdb_db.corr())


# In[57]:


#**************** We have now converted our data to usable format ******
print('Pre-processing done !!')


# In[58]:


# class_rating_ = {0:,1:,2:,3:,4:}


# In[59]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
X = imdb_db.drop(['imdb_score','class_rating'], axis=1)
y = imdb_db['class_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[60]:


X_train.shape, X_test.shape


# In[61]:


#Suppress Warninings
import warnings
warnings.filterwarnings('ignore')

#Machine Learning Algorithm (MLA) Selection and Initialization
from sklearn import ensemble, linear_model, naive_bayes, svm, tree, discriminant_analysis, gaussian_process, neighbors
from sklearn import model_selection
from xgboost import XGBClassifier

MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    
    #GLM
    linear_model.LogisticRegressionCV(),
    
    #Navies Bayes
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]



#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy Mean','MLA Train Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = pd.DataFrame()

cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .7, random_state = 0 ) 
# run model 10x with 60/30 split intentionally leaving out 10%

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X_train, y_train, cv  = cv_split, return_train_score=True, scoring='accuracy')
#     print(cv_results)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    #save MLA predictions - see section 6 for usage
    alg.fit(X_train, y_train)
    row_index+=1
    MLA_predict[MLA_name] = alg.predict(X_test)
    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare


# In[62]:


plt.close()

plt.figure(figsize=(12,8))
sb.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'g')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')


# In[63]:


model_rf = MLA[3]
model_rf.feature_importances_


# In[64]:


#Grid Search
import time
#Code is written for experimental/developmental purposes and not production ready!

#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]
grid_param = {
            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
             }

start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

start = time.perf_counter()        
best_search = model_selection.GridSearchCV(estimator = model_rf, param_grid = grid_param, cv = cv_split, scoring = 'accuracy')
best_search.fit(X_train, y_train)
run = time.perf_counter() - start

best_param = best_search.best_params_
print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(model_rf.__class__.__name__, best_param, run))
model_rf.set_params(**best_param) 


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*10)


# In[65]:


#score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
cv_results = model_selection.cross_validate(model_rf, X_train, y_train, cv  = cv_split, return_train_score=True, scoring='accuracy')
print('Train score:' , cv_results['train_score'].mean())
print('Test score:' , cv_results['test_score'].mean())


# In[66]:


pred_compare = pd.DataFrame({'Original': y_test, 'Predicted': model_rf.predict(X_test)})
pred_compare.to_csv('pred_compare.csv')


# In[67]:


from sklearn.feature_selection import RFE
#rank all features, i.e continue the elimination until the last one
rfe = RFE(model_rf, n_features_to_select=1)
rfe.fit(X_train,y_train)
 
print("Features sorted by their rank:")
feat_imp = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), X_train.columns))
print(feat_imp)


# In[68]:


#Through Recursive feature elimination, we can judge which features actually contribute to making 
#better predictions, and also reduce the dimensionallity of our dataset if lesser number of features
#give a considerate amoount of accuracy.
for how_many in [5,10,15,20,25,30,40,50]:
    X = imdb_db[[y for x,y in feat_imp[:how_many]]]
    y = imdb_db['class_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_rf.fit(X_train,y_train)
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(model_rf, X_train, y_train, cv  = cv_split, return_train_score=True, scoring='accuracy')
    print('Train score:' , cv_results['train_score'], ' : For {}'.format(how_many))
    print('Test score:' , cv_results['test_score'], ' : For {}'.format(how_many))
    print('Mean score:' , cv_results['test_score'].mean(), ' : For {} features.'.format(how_many))


# In[69]:


#We observe that just selecting 20 features for our RandomForest Model is enough to provide good accuracy on 
#cross validated scores. We keep only 20 features.
best_features = [y for x,y in feat_imp[:20]]
np.array(best_features)


# In[70]:


#Suppress Warninings
import warnings
warnings.filterwarnings('ignore')

#Machine Learning Algorithm (MLA) Selection and Initialization
from sklearn import ensemble, linear_model, naive_bayes, svm, tree, discriminant_analysis, gaussian_process, neighbors
from sklearn import model_selection
from xgboost import XGBClassifier

X = imdb_db[best_features]
y = imdb_db['class_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    
    #GLM
    linear_model.LogisticRegressionCV(),
    
    #Navies Bayes
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]



#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy Mean','MLA Train Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = pd.DataFrame()

cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .7, random_state = 0 ) 
# run model 10x with 60/30 split intentionally leaving out 10%

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X_train, y_train, cv  = cv_split, return_train_score=True, scoring='accuracy')
#     print(cv_results)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    #save MLA predictions - see section 6 for usage
    alg.fit(X_train, y_train)
    row_index+=1
    MLA_predict[MLA_name] = alg.predict(X_test)
    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare


# In[71]:


model_rf = MLA[3]


# In[72]:


#Grid Search
import time
#Code is written for experimental/developmental purposes and not production ready!

#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]
grid_param = {
            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
             }

start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

start = time.perf_counter()        
best_search = model_selection.GridSearchCV(estimator = model_rf, param_grid = grid_param, cv = cv_split, scoring = 'accuracy')
best_search.fit(X_train, y_train)
run = time.perf_counter() - start

best_param = best_search.best_params_
print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(model_rf.__class__.__name__, best_param, run))
model_rf.set_params(**best_param) 


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*10)


# In[73]:


#score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
cv_results = model_selection.cross_validate(model_rf, X_train, y_train, cv  = cv_split, return_train_score=True, scoring='accuracy')
print('Train score:' , cv_results['train_score'].mean())
print('Test score:' , cv_results['test_score'].mean())


# In[74]:


# plt.figure(figsize=(12,8))
# for x in imdb_db.columns:
#     if imdb_db[x].dtype not in [object]:
#         sb.boxplot(imdb_db[x])
#         plt.savefig(x+'_boxplot.png')
#         plt.close()


# In[75]:


imdb_db['score_predict'] = model_rf.predict(imdb_db.drop(['imdb_score','class_rating'], axis=1)[best_features])


# In[76]:


imdb_score_predict = [float(x) for x in imdb_db['score_predict']]
imdb_score_predict = np.array(imdb_score_predict)


# In[77]:


imdb_score_true = [float(x) for x in imdb_db['imdb_score']]
imdb_score_true = np.array(imdb_score_true)


# In[78]:


# [(x-y) for x,y in zip(imdb_score_true,imdb_score_predict)]


# In[79]:


from sklearn.metrics import r2_score,log_loss, accuracy_score, mean_squared_error, mean_squared_log_error, adjusted_rand_score


# In[80]:


mean_squared_log_error(imdb_score_true, imdb_score_predict), mean_squared_error(imdb_score_true, imdb_score_predict)


# In[81]:


r2_score(imdb_score_true, imdb_score_predict)


# In[82]:


imdb_db[['imdb_score', 'score_predict']].to_csv('finalimdbscore.csv')


# In[86]:


plt.title('Point point plot between actual and expected ratings')
plt.xlabel('IMDB True Score')
plt.ylabel('IMDB Predicted Score')
sb.regplot(imdb_score_true, imdb_score_predict)


# In[ ]:


#Conclusion, our model performs with an r2 score of 0.71 and a log loss of 0.008268276183748098
#The model serves as a good fit to solve the problem at hand.
#Since its a random forest model (bagging), it is very easy to deploy and the predictions can be much faster
#over boosting sequence models.

