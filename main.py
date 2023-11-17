import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import datasets
df = pd.read_csv('train.csv')

print(df.head())
print(df.info())

year = df['YR']
sns.regplot(data=df, y='LotArea', x='YrSold')
plt.show()




