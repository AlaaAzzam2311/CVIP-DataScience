import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("data.csv")
print("Head of dataset is:\n",df.head())
print("Shape of dataset is:\n",df.shape)
print("Description of dataset is:\n",df.describe())
print("Names of columns are:\n",df.columns)
print("Number of columns is:\n",len(df.columns))
print("Number of duplicated values is:\n",df.duplicated().sum())
print("Number of null values is:\n",df.isnull().sum())
print("Number of cases in each type of tumor:\n",df['diagnosis'].value_counts())
plt.figure(figsize=(12,8))
sns.barplot(x=df['diagnosis'].value_counts().index,y=df['diagnosis'].value_counts().values)
plt.title("Number of cases in each type of tumor")
plt.show()
df['diagnosis'] = df['diagnosis'].map({'B':0, 'M':1})

df_new=df.drop(['diagnosis'],axis=1)
X_train, X_test, y_train, y_test=train_test_split(df_new,df['diagnosis'],test_size=0.2,random_state=42)
model=StandardScaler()
X_train=model.fit_transform(X_train)
X_test=model.fit_transform(X_test)
logr=LogisticRegression()
logr.fit(X_train,y_train)
pred=logr.predict(X_test)
print(classification_report(y_test, pred))
acc=(accuracy_score(y_test,pred))*100
print("Accuracy of model: ",round(acc,2),"%")
