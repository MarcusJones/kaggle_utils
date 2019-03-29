#%% ===========================================================================
# Standard imports
# =============================================================================
import os
from pathlib import Path
import sys
import zipfile
import gc
import time
from pprint import pprint
from functools import reduce
from collections import defaultdict
import json
import yaml
import inspect
import h5py

#%% ===========================================================================
# Utilities
# =============================================================================
import kaggle_utils

#%% ===========================================================================
# Scientific stack
# =============================================================================
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as mpl

logging.info("{:>10}=={} as {}".format('numpy', np.__version__, 'np'))
logging.info("{:>10}=={} as {}".format('pandas', pd.__version__, 'pd'))
logging.info("{:>10}=={} as {}".format('sklearn', sk.__version__, 'sk'))
logging.info("{:>10}=={} as {}".format('matplotlib', mpl.__version__, 'mpl'))

# Load import paths for matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

# Load import paths for sklearn
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.pipeline
import sklearn.model_selection
import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.decomposition
import sklearn.compose
import sklearn.utils

#%% ===========================================================================
# Other scientific stack libraries
# =============================================================================
from sklearn_pandas import DataFrameMapper

#%% ===========================================================================
# Natural Langauge Processing
# =============================================================================
# import nltk
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from nltk.corpus import stopwords


#%% ===========================================================================
# Deep learning
# =============================================================================
# Keras
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K


# Models
import lightgbm as lgb
print("lightgbm", lgb.__version__)
import xgboost as xgb
print("xgboost", xgb.__version__)
# from catboost import CatBoostClassifier
import catboost as catb
print("catboost", catb.__version__)

# Metric
from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
