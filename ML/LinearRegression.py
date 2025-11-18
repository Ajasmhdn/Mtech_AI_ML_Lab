# ---------------------------------------------------------------
# LINEAR REGRESSION PROGRAM ON FUEL CONSUMPTION DATASET
# Includes:
#   1. Correlation Heatmap
#   2. Simple Linear Regression
#   3. Multiple Linear Regression (with One-Hot Encoding)
#   4. Polynomial Regression
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------
# 1. LOAD DATASET + CORRELATION HEATMAP
# ---------------------------------------------------------------

df = pd.read_csv("D:\FuelConsumption.csv")   # update path if needed
print(df.head())

# Heatmap for numeric columns only
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ---------------------------------------------------------------
# 2. SIMPLE LINEAR REGRESSION (Single Feature)
# Predict CO2 using FUELCONSUMPTION_CITY
# ---------------------------------------------------------------

X = df[['FUELCONSUMPTION_CITY']]
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

simple_reg = LinearRegression()
simple_reg.fit(X_train, y_train)

y_pred = simple_reg.predict(X_test)

print("\n--- SINGLE VARIABLE LINEAR REGRESSION ---")
print("Coefficient:", simple_reg.coef_[0])
print("Intercept:", simple_reg.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Plot regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, simple_reg.predict(X), color='red')
plt.xlabel("FUELCONSUMPTION_CITY")
plt.ylabel("CO2EMISSIONS")
plt.title("Simple Linear Regression")
plt.show()

# ---------------------------------------------------------------
# 3. MULTIPLE LINEAR REGRESSION (with One-Hot Encoding)
# ---------------------------------------------------------------

features = [
    'ENGINESIZE', 'CYLINDERS', 'FUELTYPE',
    'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
    'FUELCONSUMPTION_COMB'
]
target = 'CO2EMISSIONS'

X = df[features]
y = df[target]

# One-Hot Encoding for categorical feature FUELTYPE
X = pd.get_dummies(X, columns=['FUELTYPE'], drop_first=True)
print("\nEncoded Columns:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

multi_reg = LinearRegression()
multi_reg.fit(X_train, y_train)

y_pred = multi_reg.predict(X_test)

print("\n--- MULTIPLE LINEAR REGRESSION ---")
print("Coefficients:", multi_reg.coef_)
print("Intercept:", multi_reg.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted (Multiple Linear Regression)")
plt.show()

# ---------------------------------------------------------------
# 4. POLYNOMIAL REGRESSION (Degree 3)
# ---------------------------------------------------------------

X = df[['FUELCONSUMPTION_CITY']]
y = df['CO2EMISSIONS']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

y_pred = poly_reg.predict(X_test_poly)

print(f"\n--- POLYNOMIAL REGRESSION (Degree 3) ---")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Polynomial curve plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.5)

X_sorted = np.sort(X.values, axis=0)
y_poly_pred = poly_reg.predict(poly.transform(X_sorted))

plt.plot(X_sorted, y_poly_pred, color='red', linewidth=2)
plt.xlabel("FUELCONSUMPTION_CITY")
plt.ylabel("CO2EMISSIONS")
plt.title("Polynomial Regression (Degree 3)")
plt.show()