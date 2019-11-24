## FROM KAGGLE KERNEL:
import os
import random
import gc

import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn import preprocessing
# from sklearn.model_selection import KFold
# import lightgbm as lgb
# import xgboost as xgb
# import catboost as cb

# %% {"_kg_hide-input": true}
# Copy from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type
# Modified to add option to use float16 or not. feather format does not support float16.
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]):
            # skip datetime type
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
#%%

# %%
from pathlib import Path
import zipfile
DATA_PATH = '~/ashrae/data/raw'
DATA_PATH = Path(DATA_PATH)
DATA_PATH = DATA_PATH.expanduser()
assert DATA_PATH.exists(), DATA_PATH

DATA_FEATHER_PATH ='~/ashrae/data/feather'
DATA_FEATHER_PATH = Path(DATA_FEATHER_PATH)
DATA_FEATHER_PATH = DATA_FEATHER_PATH.expanduser()
assert DATA_FEATHER_PATH.exists()

# zipfile.ZipFile(DATA_PATH).infolist()

#%%
ZIPPED = False

# %%time
if ZIPPED:
    with zipfile.ZipFile(DATA_PATH) as zf:
        with zf.open('train.csv') as zcsv:
            train_df = pd.read_csv(zcsv)
        with zf.open('test.csv') as zcsv:
            test_df = pd.read_csv(zcsv)
        with zf.open('weather_train.csv') as zcsv:
            weather_train_df = pd.read_csv(zcsv)
        with zf.open('weather_test.csv') as zcsv:
            weather_test_df = pd.read_csv(zcsv)
        with zf.open('building_metadata.csv') as zcsv:
            building_meta_df = pd.read_csv(zcsv)
        with zf.open('sample_submission.csv') as zcsv:
            sample_submission = pd.read_csv(zcsv)
#%%
train_df = pd.read_csv(DATA_PATH / 'train.zip')
test_df = pd.read_csv(DATA_PATH / 'test.zip')
weather_train_df = pd.read_csv(DATA_PATH / 'weather_train.zip')
weather_test_df = pd.read_csv(DATA_PATH / 'weather_test.zip')
building_meta_df = pd.read_csv(DATA_PATH / 'building_metadata.zip')
sample_submission = pd.read_csv(DATA_PATH / 'sample_submission.zip')


# %%
# # %%time

# # Read data...
# root = '../input/ashrae-energy-prediction'

# train_df = pd.read_csv(os.path.join(root, 'train.csv'))
# weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))
# test_df = pd.read_csv(os.path.join(root, 'test.csv'))
# weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))
# building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))
# sample_submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))

# %%
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

# %% {"_kg_hide-input": true, "_kg_hide-output": true}
# # categorize primary_use column to reduce memory on merge...

# primary_use_dict = {key: value for value, key in enumerate(primary_use_list)}
# print('primary_use_dict: ', primary_use_dict)
# building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)

# gc.collect()

# %% {"_kg_hide-input": true, "_kg_hide-output": true}
reduce_mem_usage(train_df)
reduce_mem_usage(test_df)
reduce_mem_usage(building_meta_df)
reduce_mem_usage(weather_train_df)
reduce_mem_usage(weather_test_df)

# %% [markdown]
# # Save data in feather format

# %%
# %%time

train_df.to_feather('train.feather')
test_df.to_feather('test.feather')
weather_train_df.to_feather('weather_train.feather')
weather_test_df.to_feather('weather_test.feather')
building_meta_df.to_feather('building_metadata.feather')
sample_submission.to_feather('sample_submission.feather')

# %% [markdown]
# # Read data in feather format
#
# You can see "+ Add data" button on top-right of notebook, press this button and add output of this kernel, then you can use above saved feather data frame for fast loading!
#
# Let's see how fast it is.

# %%
# %%time

train_df = pd.read_feather('train.feather')
weather_train_df = pd.read_feather('weather_train.feather')
test_df = pd.read_feather('test.feather')
weather_test_df = pd.read_feather('weather_test.feather')
building_meta_df = pd.read_feather('building_metadata.feather')
sample_submission = pd.read_feather('sample_submission.feather')

# %% [markdown]
# Reduced 37.1 sec to 1.51 sec!! 😄😄😄

# %%
