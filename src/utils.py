"""
This script contains the functions for
- Preparation step:
read the data, keep the columns needed, fill missing values and scale the features.
- Training step:
- Inference step:
"""
# -----------------------------------
# libraries ----
import numpy as np
import pandas as pd
import joblib
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
# -----------------------------------
# funciones ----
# function to read the data or send a message if the file is not found
def read_data(path_to_file):
    """
    Read the data from the csv file to train the model. 
    It handles an exception if the file is not found.
    
    Parameters:
    - None
    
    Returns:
    - df_train: DataFrame
        Dataframe with the data to train the model"""
    try:
        df_train = pd.read_csv(path_to_file)
        return df_train
    except FileNotFoundError:
        print("File not found")
        return None
    
# function to keep columns needed for data training
def keep_columns(df_train):
    """
    Keep only the columns that are going to be used to train the model. 
    The columns are selected based on the data exploration.
    
    Parameters:
    - df_train: DataFrame
        Dataframe with the data to train the model
    
    Returns:
    - df_train_filt: DataFrame
        Dataframe with the selected columns"""
    columns_to_keep = [
        'GrLivArea', 'LotArea',
        'YearBuilt', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars',
        'GarageArea', 'Fireplaces', 'SalePrice'
    ]
    df_train_filt = df_train[columns_to_keep]
    return df_train_filt

# function to fill missing values with average per column
def fill_missing_values(df_train_filt):
    """
    Fill the missing values with the mean of each column.
    
    Parameters:
    - df_train_filt: DataFrame
        Dataframe with the selected columns
    
    Returns:
    - df_train_filt: DataFrame
        Dataframe with the missing values filled"""
    mean_values = df_train_filt.mean()
    df_train_filt = df_train_filt.fillna(mean_values)
    return df_train_filt

# function to scale the features
def scale_train(df_train_filt):
    """
    Scale the features to be used in the model. 
    
    Parameters:
    - df_train_filt: DataFrame
        Dataframe with the selected columns
    
    Returns:
    - X_train_scaled: array
        Array with the scaled features"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = df_train_filt.drop("SalePrice", axis=1)
    y_train = df_train_filt["SalePrice"]
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, y_train, scaler

# function to train the model
def train_model(X_train_scaled, y_train, min=1, max=31):
    """
    Train the model using the training data. The model is a KNN regressor.
    
    Parameters:
    - X_train_scaled: DataFrame
        Dataframe with the scaled features
    - y_train: DataFrame
        Dataframe with the target variable
    - min: int
        Minimum number of neighbors to consider
    - max: int
        Maximum number of neighbors to consider
    
    Returns:
    - knn_best: KNeighborsRegressor
        Trained model"""
    
    # declare the model
    knn_regressor = KNeighborsRegressor()

    # grid search to find the best number of neighbors
    param_grid = {'n_neighbors': np.arange(min, max)}

    # train the model
    grid_search = GridSearchCV(knn_regressor, param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)

    # get the best model
    best_k = grid_search.best_params_['n_neighbors']
    knn_best = grid_search.best_estimator_

    return knn_best

# function to save the model
def save_model(knn_best, path_to_save):
    """
    Save the trained model in a file.
    
    Parameters:
    - knn_best: KNeighborsRegressor
        Trained model
    - path_to_save: str
        Path to save the model"""
    joblib.dump(knn_best, path_to_save)