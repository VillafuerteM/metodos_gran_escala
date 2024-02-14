"""
Inference
This script aims to make predictions using the trained model.
"""
# ------------------------------------
# librerias
import pandas as pd
import joblib
import src.utils.py as utils
# ------------------------------------
# funciones ----
# las funciones se importan de src.utils.py

# ------------------------------------
# main ----
# cargar el modelo, el scaler y los datos de prueba
knn_best = utils.load_model('artifacts/knn_best_model.joblib')
scaler = utils.load_scaler('artifacts/scaler.joblib')
x_test = pd.read_csv('data/raw/x_test.csv')

# escalar las features
x_test_scaled = utils.scale_test(x_test, scaler)

# hacer las predicciones
y_pred = utils.make_predictions(knn_best, x_test_scaled)

# guardar las predicciones
utils.save_predictions(y_pred, 'data/inferences/y_pred.csv')
