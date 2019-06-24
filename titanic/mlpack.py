# %load mlpack.py
# %load mlpack.py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sb
from sklearn.metrics import accuracy_score,r2_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

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
# %load mlpack.py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sb
from sklearn.metrics import accuracy_score,r2_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

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

	return df