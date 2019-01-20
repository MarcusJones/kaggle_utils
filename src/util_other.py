
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import seaborn as sns

#--- Loading tools
#%% Write the CSV dataframes to HDF5
def save_df_hd5(dfs,file_dict):
    raise "Pasted from Spyder, clean up"
    for file in dfs:
        #dfs[file]
        file_path = file_dict[file]
        fname,_ = os.path.splitext(file_path)
        fname,_ = os.path.splitext(fname)
        fname += ".h5"
        logging.debug("Writing {}, {} rows, cols, to {}".format(file,dfs[file].shape,fname))
        dfs[file].to_hdf(fname, 'data', format='table')

#--- Feature engineering
#%% Aggregator for numeric
def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    
    # First calculate counts
    counts = pd.DataFrame(df.groupby(group_var, as_index = False)[df.columns[1]].count()).rename(columns = {df.columns[1]: '%s_counts' % df_name})
    
    # Group by the specified variable and calculate the statistics
    agg = df.groupby(group_var).agg(['mean', 'max', 'min', 'sum']).reset_index()
    
    # Need to create new column names
    columns = [group_var]
    
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
              
    #  Rename the columns
    agg.columns = columns
    
    # Merge with the counts
    agg = agg.merge(counts, on = group_var, how = 'left')
    
    return agg


def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical

#--- EDA tools

# Function to calculate correlations with the target for a dataframe
def target_corrs(df):

    # List of correlations
    corrs = []

    # Iterate through the columns 
    for col in df.columns:
        print(col)
        # Skip the target column
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df[col])

            # Append the list as a tuple
            corrs.append((col, corr))
            
    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    
    return corrs


#%% Create a summary of each dataframe and their columns
def save_summaries_excel():
    raise "Pasted from Spyder, clean up"
    df_summaries = dict()
    for df_name in dfs:
        df_summaries[df_name] = exergyml_util_other.create_column_summary(dfs[df_name])
        #print(k)

#%% Print summaries to excel
    PATH_PROJECT_ROOT
    excel_file_name = "all columns r00.xlsx"
    
    with pd.ExcelWriter(os.path.join(PATH_PROJECT_ROOT,excel_file_name), engine='xlsxwriter') as writer:
        for df_name in df_summaries:
            df_summaries[df_name].to_excel(writer, sheet_name=df_name)
        #writer.save()



#%% Function to calculate missing values by column# Funct 
def missing_values_table(df, sort = True, show_all=True):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        if sort:
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


def create_column_summary(this_df):
    
    #this_df = dfs['bureau_balance']
    #this_df.columns
    
    this_summary = exergyml_util_other.missing_values_table(this_df,sort=False)
    this_summary = pd.concat([this_summary, this_df.dtypes], axis=1)
    
    this_summary = this_summary.rename({0:"dtype"},axis='columns')
    
    this_summary['num_cats'] = None
    this_summary['cats'] = ""
    for cat_col in this_df.select_dtypes(include='category'):
        this_summary.loc[cat_col, 'num_cats'] = this_df[cat_col].nunique()
        
        these_categories = this_df[cat_col].cat.categories.values.tolist()
        categories_str = ", ".join(these_categories)
        this_summary.loc[cat_col, 'cats'] = categories_str
        
        #this_df.select_dtypes(include='category').nunique()
    logging.debug("Created summary for a data frame")
    return this_summary




def grid_scores_to_df(grid_scores):
    """
    Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to a tidy
    pandas DataFrame where each row is a hyperparameter-fold combinatination.
    """
    rows = list()
    for i,grid_score in enumerate(grid_scores):
        
        for fold, score in enumerate(grid_score.cv_validation_scores):
            #row = dict()
            row = grid_score.parameters.copy()
            row['param_set'] = i
            row['fold'] = fold
            row['score'] = score
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

#--- Conversions
def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

def convert_categorical(df):
    logging.debug("\tConverted object columns to category")
    return pd.concat([
        df.select_dtypes(include=[], exclude=['object']),
        df.select_dtypes(include=['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1).reindex(df.columns, axis=1)

#--- Plotting

#%% Plot correlations
# Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):
    
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_repaid = df.loc[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)




#exergy.util_excel
def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours





# http://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return mpl.colors.LinearSegmentedColormap('colormap',cdict,1024)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          size = None
                          
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    fig, ax = plt.subplots(figsize=size)         # Sample figsize in inches



    plt.style.use('ggplot')


#sns.heatmap(addr_cross_cat, linewidths=.5,cmap='jet');
#sns.heatmap(addr_cross_cat, linewidths=.5,cmap = cmap,ax=ax)
#plt.tight_layout(pad=5)
#plt.suptitle(title_str,fontname = 'Arial', fontsize=16)
#plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout(pad=5)

#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")





