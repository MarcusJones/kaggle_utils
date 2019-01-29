import sklearn as sk
import pandas as pd
import time
import numpy as np
import sklearn.preprocessing
import sklearn.base
from sklearn.base import BaseEstimator, TransformerMixin

#%%
def timeit(method):
    """ Decorator to time execution of transformers
    :param method:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("\t {} {:2.1f}s".format(method.__name__, (te - ts)))
        return result
    return timed

#%%
class TransformerLog():
    """Add a .log attribute for logging
    """
    @property
    def log(self):
        return "Transformer: {}".format(type(self).__name__)

#%%
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        assert isinstance(df, pd.DataFrame)
        return df.select_dtypes(include=[self.dtype])

# %%

class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True, name=None):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector
        self.name = name

        # Ensure columns in a list()
        if isinstance(self.columns, str):
            self.columns = [self.columns]

        logging.info("Init {} on cols: {}".format(name, columns))

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError(
                    "Keys are missing in the record: {}, columns required:{}".format(missing_columns_, self.columns))

    def transform(self, x):
        logging.info("{} is transforming...".format(self.name))
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError("Input is not a pandas DataFrame it's a {}".format(type(x)))

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return list(x[selected_cols[0]])
        else:
            return x[selected_cols]


# %%==============================================================================
# Empty
# ===============================================================================
class Empty(BaseEstimator, TransformerMixin, TransformerLog):
    """An empty transformer
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        print(self.log)
        return df

# %%==============================================================================
# Imputer1D - Simple Imputer wrapper
# ===============================================================================
class Imputer1D(sk.preprocessing.Imputer):
    """
    A simple wrapper class on Imputer to avoid having to make a single column 2D.
    """
    def fit(self, X, y=None):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
        # Call the Imputer as normal, return result
        return super(Imputer1D, self).fit(X, y=None)

    def transform(self, X, y=None):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
            # Call the Imputer as normal, return result
        return super(Imputer1D, self).transform(X)

#%%
class MultipleToNewFeature(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """
    def __init__(self, selected_cols, new_col_name,func):
        self.selected_cols = selected_cols
        self.new_col_name = new_col_name
        self.func = func

    def fit(self, X, y=None):
        return self
    @timeit
    def transform(self, df, y=None):
        # print(df)
        df[self.new_col_name] = df.apply(self.func, axis=1)
        print(self.log, "{}({}) -> ['{}']".format(self.func.__name__,self.selected_cols,self.new_col_name))
        return df


#%%
class NumericalToCat(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """Convert numeric indexed column into dtype category with labels
    Convert a column which has a category, presented as an Integer
    Initialize with a dict of ALL mappings for this session, keyed by column name
    (This could be easily refactored to have only the required mapping)
    """

    def __init__(self, label_map_dict, allow_more_labels=False):
        self.label_map_dict = label_map_dict
        self.allow_more_labels = allow_more_labels

    def fit(self, X, y=None):
        return self

    def get_unique_values(self, this_series):
        return list(this_series.value_counts().index)

    def transform(self, this_series):
        if not self.allow_more_labels:
            if len(self.label_map_dict) > len(this_series.value_counts()):
                msg = "{} labels provided, but {} values in column!\nLabels:{}\nValues:{}".format(
                    len(self.label_map_dict), len(this_series.value_counts()), self.label_map_dict,
                    self.get_unique_values(this_series), )
                raise ValueError(msg)

        if len(self.label_map_dict) < len(this_series.value_counts()):
            raise ValueError

        assert type(this_series) == pd.Series
        # assert this_series.name in self.label_map_dict, "{} not in label map!".format(this_series.name)
        return_series = this_series.copy()
        # return_series = pd.Series(pd.Categorical.from_codes(this_series, self.label_map_dict))
        return_series = return_series.astype('category')
        return_series.cat.rename_categories(self.label_map_dict, inplace=True)
        # print(return_series.cat.categories)

        assert return_series.dtype == 'category'
        return return_series


# %%==============================================================================
# NumericalToCat - DataFrameMapper only!
# ===============================================================================
class NumericalToCatOld(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """Convert numeric indexed column into dtype category with labels
    Convert a column which has a category, presented as an Integer
    Initialize with a dict of ALL mappings for this session, keyed by column name
    (This could be easily refactored to have only the required mapping)
    """
    def __init__(self,label_map_dict):
        self.label_map_dict = label_map_dict

    def fit(self, X, y=None):
        return self

    def transform(self, this_series):
        assert type(this_series) == pd.Series
        assert this_series.name in self.label_map_dict, "{} not in label map!".format(this_series.Name)
        return_series = this_series.copy()
        return_series = pd.Series(pd.Categorical.from_codes(this_series, self.label_map_dict))
        # return_series = return_series.astype('category')
        # return_series.cat.rename_categories(self.label_map_dict[return_series.name], inplace=True)
        print(self.log, return_series.cat.categories)

        assert return_series.dtype == 'category'
        return return_series

# r  = df_trf.Vaccinated
# r = r.astype('category')
# r.cat.rename_categories(label_maps['Vaccinated'],inplace=True)

# train['FurLength'] = train['FurLength'].astype('category')
# train['FurLength'].cat.rename_categories(map_FurLength,inplace=True)

#%%==============================================================================
# WordCounter
# ===============================================================================
class WordCounter(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """ Count the words in the input column
    """
    def __init__(self, col_name, new_col_name):
        self.col_name = col_name
        self.new_col_name = new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        new_col = df[self.col_name].apply(lambda x: len(x.split(" ")))
        df[self.new_col_name] = new_col
        print(self.log, self.new_col_name)
        return df

#%% =============================================================================
# ConvertToDatetime
# ===============================================================================
class ConvertToDatetime(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """

    def __init__(self, time_col_name, unit='s'):
        self.time_col_name = time_col_name
        self.unit = unit

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df[self.time_col_name] = pd.to_datetime(df[self.time_col_name], unit=self.unit)
        print("Transformer:", type(self).__name__, "converted", self.time_col_name, "to dt")
        return df

#%% =============================================================================
# TimeProperty
# ===============================================================================
class TimeProperty(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """
    def __init__(self, time_col_name, new_col_name, time_property):
        """

        :param time_col_name: Source column, MUST BE A datetime TYPE!
        :param new_col_name: New column name
        :param time_property: hour, month, dayofweek
        """
        self.time_col_name = time_col_name
        self.new_col_name = new_col_name
        self.time_property = time_property

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        original_shape = df.shape
        if self.time_property == 'hour':
            df[self.new_col_name] = df[self.time_col_name].dt.hour
        elif self.time_property == 'month':
            df[self.new_col_name] = df[self.time_col_name].dt.month
        elif self.time_property == 'dayofweek':
            df[self.new_col_name] = df[self.time_col_name].dayofweek
        else:
            raise
        print("Transformer:", type(self).__name__, original_shape, "->", df.shape, vars(self))
        return df
# Debug:
# df = X_train
# time_col_name = 'question_utc'
# new_col_name = 'question_hour'
# time_property = 'hour'
# time_col_name = 'question_utc'
# new_col_name = 'question_month'
# time_property = 'month'
# time_adder = TimeProperty(time_col_name,new_col_name,time_property)
# res=time_adder.transform(df)

#%% =============================================================================
# DEPRECIATED - AnswerDelay
# ===============================================================================
class AnswerDelay(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """ Used once, not general, gets time elapsed
    """

    def __init__(self, new_col_name, divisor=1):
        self.new_col_name = new_col_name
        self.divisor = divisor

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df[self.new_col_name] = df['answer_utc'] - df['question_utc']
        df[self.new_col_name] = df[self.new_col_name].dt.seconds / self.divisor
        print(self.log)
        return df


# Debug:
# df = X_train
# new_col_name = 'answer_delay_seconds'
# answer_delay_adder = AnswerDelay(new_col_name)
# res=answer_delay_adder.transform(df)

#%% =============================================================================
# ValueCounter
# ===============================================================================
class ValueCounter(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """??
    """

    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        # Count the number of unique entries in a column
        # reset_index() is used to maintain the DataFrame for merging
        selected_df_col = df[self.col_name].value_counts().reset_index()
        # Create a new name for this column
        selected_df_col.columns = [self.col_name, self.col_name + '_counts']
        print(self.log)
        return pd.merge(selected_df_col, df, on=self.col_name)

# %%=============================================================================
# DEPRECIATED ConvertDoubleColToDatetime
# ===============================================================================
class ConvertDoubleColToDatetime(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """

    # pd.options.mode.chained_assignment = None  # default='warn'
    def __init__(self, new_col_name, name_col1, name_col2, this_format):
        self.new_col_name = new_col_name
        self.name_col1 = name_col1
        self.name_col2 = name_col2
        self.this_format = this_format

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        combined_date_string_series = df.loc[:, self.name_col1] + " " + df.loc[:, self.name_col2]
        with ChainedAssignment():
            df.loc[:, self.new_col_name] = pd.to_datetime(combined_date_string_series, format=self.this_format)
        #        pd.options.mode.chained_assignment = 'warn'  # default='warn'

        # print("Transformer:", type(self).__name__, "converted", self.new_col_name, "to dt")
        print(self.log)
        return df

# Debug:
# df = sfpd_head
# new_col_name = 'dt'
# time_adder = ConvertDoubleColToDatetime(new_col_name,name_col1="Date", name_col2="Time",this_format=r'%m/%d/%Y %H:%M')
# res=time_adder.transform(df)