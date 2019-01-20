import utm

#%%=============================================================================
# UTM Grid 
#===============================================================================
class UTMGridConvert(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    def __init__(self, new_col_name, lat_col, long_col):
        self.new_col_name = new_col_name
        self.lat_col = lat_col
        self.long_col = long_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, df ):
        with ChainedAssignment():
            df.loc[:,self.new_col_name]=df.apply(lambda row: utm.from_latlon(row[self.lat_col], row[self.long_col]), axis=1)
        print(self.log)
        return df