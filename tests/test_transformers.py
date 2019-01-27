#%%
#  import pytest

import sklearn as sk
import pandas as pd
import time
import numpy as np

from kaggle_utils import transformers as trf

def test_empty():
    this_trf = trf.Empty()


import sklearn.datasets

#%%
this_bunch = sk.datasets.load_boston()

df_data = pd.DataFrame(this_bunch['data'], columns=this_bunch['feature_names'])
ser_tgt = pd.Series(this_bunch['target'])
pd.concat([df_data , ser_tgt])
this_bunch['feature_names']
#%%

df = pd.DataFrame

arr = np.array([5,5])
n_records = 10
n_cols = 5
col_names = ["Col "+str(i) for i in range(n_cols)]
df = pd.DataFrame(np.random.randint(0,10,(n_records,n_cols)),columns=col_names)
# Add some string cols
# Add some categorical columns
# Add some date time cols, both in raw string, and in already-converted

pd.DataFrame
