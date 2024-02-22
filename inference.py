"""
Inference
This script aims to make predictions using the trained model.
"""
# ------------------------------------
# librerias
import pandas as pd
import argparse
import os
from src import utils
# ------------------------------------
# funciones ----
# las funciones se importan de src.utils.py

# ------------------------------------
# agruments ----
parser = argparse.ArgumentParser()
parser.add_argument('test_data', help='name of the test data file/ new data to predict')
args = parser.parse_args()

# -----------------------------------
# main ----
# cargar el modelo, el scaler y los datos de prueba
knn_best = utils.load_model("data/artifacts/knn_best.joblib")
scaler = utils.load_scaler("data/artifacts/scaler.joblib")
x_test = pd.read_csv(os.path.join("data", "raw", args.test_data))

# seleccionar las columnas
x_test_filt = utils.keep_columns_test(x_test)

# lidiar con los missing values
x_test_filt = utils.fill_missing_values_test(x_test_filt)

# escalar las features
x_test_scaled = utils.scale_test(x_test_filt, scaler)

# hacer las predicciones
y_pred = utils.make_predictions(knn_best, x_test_scaled)

# guardar las predicciones
utils.save_predictions(y_pred, 'data/inferences/y_pred.csv')
