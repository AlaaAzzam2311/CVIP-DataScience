import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df=pd.read_csv("spam.csv",encoding='latin1')
print("Head of dataset:\n",df.head())
print("Shape of dataset:\n",df.shape)
print("Names of columns:\n",df.columns)
print("Number of columns:\n",len(df.columns))
print("Number of null values in each column:\n",df.isnull().sum())
print("Number of duplicated values:\n",df.duplicated().sum())
plt.figure(figsize=(12,8))
sns.barplot(x=df['v1'].value_counts().index,y=df['v1'].value_counts().values)
plt.title("Number of messages for each type of classification")
plt.show()

df['v1'] = df['v1'].map({'ham':0, 'spam':1})
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2,random_state=42)
model=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
model.fit(X_train,y_train)
pred=model.predict(X_test)
print(classification_report(y_test, pred))
acc=(accuracy_score(y_test,pred))*100
print("Accuracy of model: ",round(acc,2),"%")
