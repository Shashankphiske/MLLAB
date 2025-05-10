import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns 

data = pd.read_csv(".csv")
feature_cols = ["age", "pregnant", "insulin", "bmi"]
x = data[feature_cols]
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)