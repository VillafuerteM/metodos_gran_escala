"""
This script contains the functions for
- Preparation step:
read the data, keep the columns needed, fill missing values and scale the features.
- Training step:
train the model using the training data and save the model.
- Inference step:
load the model and make predictions.
"""
# -----------------------------------
# libraries ----
import numpy as np
import pandas as pd
import joblib
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
    It handles the error in which the columns are not found and sends 
    a message to the user stating that the columns are not found.
    
    Parameters:
    - df_train: DataFrame
        Dataframe with the data to train the model
    
    Returns:
    - df_train_filt: DataFrame
        Dataframe with the selected columns"""
    try: 
        columns_to_keep = [
            'GrLivArea', 'LotArea',
            'YearBuilt', 'FullBath', 'HalfBath',
            'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars',
            'GarageArea', 'Fireplaces', 'SalePrice'
        ]
        df_train_filt = df_train[columns_to_keep]
        return df_train_filt
    except KeyError:
        print("Columns not found")
        return None

# function to keep columns needed for data testing
def keep_columns_test(df_test):
    """
    Keep only the columns that are going to be used to predict with the model. 
    The columns are selected based on the data exploration.
    It tries to keep the columns and if it fails, it sends a message to the user.
    
    Parameters:
    - df_train: DataFrame
        Dataframe with the data to train the model
    
    Returns:
    - df_train_filt: DataFrame
        Dataframe with the selected columns"""
    try:
        columns_to_keep = [
            'GrLivArea', 'LotArea',
            'YearBuilt', 'FullBath', 'HalfBath',
            'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars',
            'GarageArea', 'Fireplaces'
        ]
        df_test_filt = df_test[columns_to_keep]
        return df_test_filt
    except KeyError:
        print("Columns not found")
        return None

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

# function to fill missing values in test with 0
def fill_missing_values_test(df_test_filt):
    """
    Fill the missing values with 0.
    
    Parameters:
    - df_test_filt: DataFrame
        Dataframe with the selected columns
    
    Returns:
    - df_test_filt: DataFrame
        Dataframe with the missing values filled"""
    df_test_filt = df_test_filt.fillna(0)
    return df_test_filt

# function to scale the features
def scale_train(df_train_filt):
    """
    Scale the features to be used in the model. 
    
    Parameters:
    - df_train_filt: DataFrame
        Dataframe with the selected columns
    
    Returns:
    - x_train_scaled: array
        Array with the scaled features"""
    scaler = StandardScaler()
    x_train = df_train_filt.drop("SalePrice", axis=1)
    y_train = df_train_filt["SalePrice"]
    x_train_scaled = scaler.fit_transform(x_train)
    return x_train_scaled, y_train, scaler

# function to train the model
def train_model(x_train_scaled, y_train, minimo=1, maximo=31):
    """
    Train the model using the training data. The model is a KNN regressor.
    
    Parameters:
    - x_train_scaled: DataFrame
        Dataframe with the scaled features
    - y_train: DataFrame
        Dataframe with the target variable
    - minimo: int
        Minimum number of neighbors to consider
    - maximo: int
        Maximum number of neighbors to consider
    
    Returns:
    - knn_best: KNeighborsRegressor
        Trained model"""

    # declare the model
    knn_regressor = KNeighborsRegressor()

    # grid search to find the best number of neighbors
    param_grid = {'n_neighbors': np.arange(minimo, maximo)}

    # train the model
    grid_search = GridSearchCV(knn_regressor, param_grid, cv=5)
    grid_search.fit(x_train_scaled, y_train)

    # get the best model
    knn_best = grid_search.best_estimator_

    return knn_best

# function to save the model
def save_model(knn_best, path_to_save):
    """
    Save the trained model in a file. Tries to save the model to path
    and if it fails, it sends a message to the user.
    
    Parameters:
    - knn_best: KNeighborsRegressor
        Trained model
    - path_to_save: str
        Path to save the model"""
    try:
        joblib.dump(knn_best, path_to_save)
    except:
        print("Model not saved, path not found")

# function to load the model
def load_model(path_to_model):
    """
    Load the trained model from a file. Tries to load the model from path
    and if it fails, it sends a message to the user.
    
    Parameters:
    - path_to_model: str
        Path to load the model
    
    Returns:
    - knn_best: KNeighborsRegressor
        Trained model"""
    try:
        knn_best = joblib.load(path_to_model)
        return knn_best
    except:
        print("Model not loaded, path not found")
        return None

# load scaler
def load_scaler(path_to_scaler):
    """
    Load the scaler used to scale the features.
    
    Parameters:
    - path_to_scaler: str
        Path to load the scaler
    
    Returns:
    - scaler: StandardScaler
        Scaler used to scale the features"""
    scaler = joblib.load(path_to_scaler)
    return scaler

# function to scale the features
def scale_test(df_test, scaler):
    """
    Scale the features to be used in the model. 
    
    Parameters:
    - df_test: DataFrame
        Dataframe with the selected columns
    - scaler: StandardScaler
        Scaler used to scale the features
    
    Returns:
    - x_test_scaled: array
        Array with the scaled features"""
    x_test = df_test
    x_test_scaled = scaler.transform(x_test)
    return x_test_scaled

# function to make predictions
def make_predictions(knn_best, x_test_scaled):
    """
    Make predictions using the trained model.
    
    Parameters:
    - knn_best: KNeighborsRegressor
        Trained model
    - x_test_scaled: array
        Array with the scaled features
    
    Returns:
    - predicted_sale_price: array
        Array with the predicted sale prices"""
    predicted_sale_price = knn_best.predict(x_test_scaled)
    return predicted_sale_price

# function to save the predictions
def save_predictions(y_pred, path_to_save):
    """
    Save the predictions in a file.
    
    Parameters:
    - y_pred: array
        Array with the predicted sale prices
    - path_to_save: str
        Path to save the predictions"""
    pd.DataFrame(y_pred).to_csv(path_to_save, index=False)
