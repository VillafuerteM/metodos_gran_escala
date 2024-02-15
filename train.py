"""
Model training
Train the model using the training data. The model is a KNN regressor.
"""

# -----------------------------------
# libraries ----
from src import utils

# -----------------------------------
# funciones ----
# functions are declared in utils.py file located in src folder

# -----------------------------------
# main ----
# Read the data
X_train_scaled = utils.read_data("data/prep/X_train_scaled.csv")
y_train = utils.read_data("data/prep/y_train.csv")

# Train and get best KNN model
knn_best = utils.train_model(X_train_scaled, y_train, minimo=1, maximo=31)

# Save the model as joblib file
utils.save_model(knn_best, "data/artifacts/knn_best.joblib")
