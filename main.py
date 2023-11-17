import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import datasets
df = pd.read_csv('train.csv')
pd.set_option('display.max_columns', None)

# Data verkenning
# print(df.describe())

# Correlatie Matrix
# corr_matrix = df.corr()
# print(corr_matrix)
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f')
# plt.show()


# sns.histplot(df['SalePrice'], bins=30, kde=True)
# plt.show()

# Missende gegevens
# Controleer voor missende gegevens
missing_data = df.isnull().sum()
# print(missing_data[missing_data > 0])

# Data aanvullen
df['Alley'] = df['Alley'].fillna(0)
alley = {'Grvl': 2, 'Pave': 1, 0: 0}
df['Alley'] = df['Alley'].map(alley)

df['LotFrontage'] = df['LotFrontage'].fillna(0)

# Vult NaN waardes met 0
df['MasVnrType'] = df['MasVnrType'].fillna(0)
# Dit is een dictionary met de text waardes (Zie data description) en hun numerieke waardes
brick_type = {'BrkCmn': 5, 'BrkFace': 4, 'CBlock': 3, 'None': 2, 'Stone': 1, 0: 0}
# Dit mapt automatisch de numerieke waardes aan de tekst waardes
df['MasVnrType'] = df['MasVnrType'].map(brick_type)
# Controle of de map goed is uitgevoerd. Mag geen NaN meer bevatten!
print(df['MasVnrType'].unique())
# Deze code controleert of alle lege waardes zijn vervangen met numerieke waardes.
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])




