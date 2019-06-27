#Suppress Warninings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sb
from sklearn.metrics import accuracy_score,r2_score,classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold,StratifiedKFold, cross_val_score
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

def read_basic(name=None):
	if(name==None):
		raise ValueError('No File Name provided')
		return 0

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
