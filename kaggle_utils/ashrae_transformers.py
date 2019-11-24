from sklearn.base import BaseEstimator, TransformerMixin


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self._feature_names]
        except KeyError:
            cols_error = list(set(self._feature_names) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class TemporalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self._column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert pd.api.types.is_datetime64_any_dtype(X[self._column])

        out = pd.DataFrame()

        out['hour'] = X[self._column].dt.hour
        out['month'] = X[self._column].dt.month
        # X['week'] = X[self._column].dt.week
        out['weekday'] = X[self._column].dt.weekday
        out['quarter'] = X[self._column].dt.quarter
        out['weekend'] = np.where(X[self._column].dt.weekday > 4, 1, 0)
        return out
