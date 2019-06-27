from dask import delayed, compute
import dask
import pandas as pd
import requests
import numpy as np
import json

from dask.diagnostics import ProgressBar
ProgressBar().register()

# AS per the second dataset on Kaggle, the block code information can be obtained using the API:
# https://geo.fcc.gov/api/census/#!/block/get_block_find. We use this to find the appropraiate block code
# for all lat,long pairs in the NYC dataset.

class PredictionNYC:
	def read_basic(self, name=None):
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

		self.nyc_df = df.copy()

		self.lats = self.nyc_df.latitude.values
		self.longs = self.nyc_df.longitude.values

	@delayed
	def fips_code(self, index_):
		url = 'https://geo.fcc.gov/api/census/block/find?latitude={}&longitude={}&format=json'.format(self.lats[index_],self.longs[index_])
		r = str(requests.get(url = url).content)
	#     print(r)
	#     print(r.find('FIPS'))
		return np.int(r[r.find('FIPS')+7:r.find("bbox")-3])


nyc_pred = PredictionNYC()
nyc_pred.read_basic('dataset_TSMC2014_NYC.csv')

delayed_results = [nyc_pred.fips_code(x) for x in range(nyc_pred.nyc_df.shape[0])]

with ProgressBar():
	fips_codes = compute(*delayed_results)