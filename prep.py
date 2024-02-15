"""
Data preparation
This script aims to prepare the data for training the model. It first reads the data,
keeps only the columns needed, fills missing values and scales the features. The 
resulting data is saved in the data/prep folder. The scaler is saved in the same folder 
for furtger use in the inference step.
"""

# -----------------------------------
# libraries ----
import src.utils as utils
import pandas as pd
import joblib

# -----------------------------------
# funciones ----
# functions are declared in utils.py file located in src folder

# -----------------------------------
# main ----
# read the data
df_train = utils.read_data("data/raw/train.csv")

# keep only the columns needed
df_train_filt = utils.keep_columns(df_train)

# fill missing values
df_train_filt = utils.fill_missing_values(df_train_filt)

# scale the features
X_train_scaled, y_train, scaler = utils.scale_train(df_train_filt)

# save the data
pd.DataFrame(X_train_scaled).to_csv("data/prep/X_train_scaled.csv", index=False)
pd.DataFrame(y_train).to_csv("data/prep/y_train.csv", index=False)

# save the scaler
joblib.dump(scaler, "data/artifacts/scaler.joblib")