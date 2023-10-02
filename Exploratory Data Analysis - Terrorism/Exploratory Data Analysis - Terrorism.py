import pandas as pd
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
df= pd.read_csv("globalterrorismdb_0718dist.csv", encoding='latin1')
print("First 5 rows are:\n",df.head())
print("Names of columns are:\n",df.columns)
print("Number of columns is:\n",len(df.columns))
print("Shape of data is:\n",df.shape)
print("Information about the data:")
print(df.info())
print("Description of data is:\n",df.describe())
print("Null values:\n",df.isnull().sum())
print("Number of duplicated values is:\n",df.duplicated().sum())
print("Data types of values in each column:\n",df.dtypes)
print("Total number of victims is:\n",df['country'].values.sum())
df_countries=df.groupby(["country_txt"], as_index=False)["country"].count()
df_countries=df_countries.sort_values('country', ascending=False)
print("Number of victims in each country:\n",df_countries)
plt.subplots(figsize=(12,8))
sns.barplot(x=df['attacktype1_txt'].value_counts().index[:10], y=df['attacktype1_txt'].value_counts().values[:10])
plt.title("Number of attacks for each type of attack")
plt.xticks(rotation=10)
plt.show()
attacked_cities=df.city.values
print("Names of attacked cities are:\n",attacked_cities)
plt.subplots(figsize=(12, 8))
sns.barplot(x=df_countries['country_txt'].values[:10],y=df_countries['country'].values[:10])
plt.title('Top countries Affected')
plt.xticks(rotation=10)
plt.show()
terrorism_group=df.groupby(["gname"], as_index=False)["gname"].value_counts().values
print("Number of attacks for each terrorism group:\n",terrorism_group)
