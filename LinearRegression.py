import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

x = np.array([])
y = np.array([])

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
y_pred = model.predict(x)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)