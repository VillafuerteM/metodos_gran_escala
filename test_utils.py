"""
This script contains the functions for testing the functions in utils.py file.
"""
# -----------------------------------
# libraries ----
import numpy as np
import pandas as pd
import joblib
import unittest
from utils import read_data
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
# -----------------------------------
# funciones ----
