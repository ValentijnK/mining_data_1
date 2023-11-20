import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

# import datasets
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
# Display Options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Voorbeeld van de data bekijken
print(df.head())
# Dataset variabelen verkennen
print(df.info())

# Data opschonen
df = df.fillna(0)
# Selecteer alle niet numerieke kolommen
non_numeric_cols = df.select_dtypes(exclude='number')
# Maak voor elke kolom een value mapping naar numeriek. (Benodigd voor random forest algoritme)
for column in non_numeric_cols.columns:
    # Maak een lege dict aan voor de value mapping
    value_mapping = {}
    # Geef elke unieke waarde in de kolom een numerieke waarde
    for unique_value in df[column].unique():
        value_mapping[unique_value] = len(value_mapping) + 1
    # Map de numerieke waarde aan de kolom waarde
    df[column] = df[column].map(value_mapping)

# Correlatie heatmap maken
corr_matrix = df.corr()
plt.figure(figsize=(40, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.show()




# Subset dataset voor regressie model
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Verdeel de data in train en test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# init lineair regressie model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit het model op de trainingsdata
model.fit(X_train, Y_train)

# Maak voorspellingen op de test set
y_pred = model.predict(X_test)

# Bereken en print score
model.score(X_train, Y_train)
score = round(model.score(X_train, Y_train) * 100, 2)
print(f'Score: {score}')

# Maak plot voor de regressielijn
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')

sort_order = np.argsort(X_test.values.flatten())
plt.plot(X_test.values[sort_order], y_pred[sort_order], color='blue', linewidth=3, label='Regression Line')

plt.title('Random Forest Regression')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.show()

# Laat de meest interessante waardes zien:
importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(model.feature_importances_,3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')

# Residuen plot
residu = Y_test - y_pred
plt.scatter(y_pred, residu, color='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residuen Plot')
plt.xlabel('GrLivArea')
plt.ylabel('Residuen')
plt.show()

print(importances.head(80))