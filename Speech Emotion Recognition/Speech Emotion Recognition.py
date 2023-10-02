import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
import librosa
import librosa.display
from IPython.display import Audio
from IPython.core.display import display
from playsound import playsound
#from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


Tess="C:/Users/User/PycharmProjects/CVIP golden_task2/Tess"
emotion_df=[]
for dir in os.listdir(Tess):
    for wav in os.listdir(os.path.join(Tess, dir)):
     info = wav.partition(".wav")[0].split("_")
     if info[2] == 'sad':
      emotion_df.append(("sad", os.path.join(Tess,dir, wav)))
     elif info[2] == 'angry':
      emotion_df.append(("angry",os.path.join(Tess,dir, wav)))
     elif info[2] == 'disgust':
      emotion_df.append(("disgust", os.path.join(Tess,dir, wav)))
     elif info[2] == 'fear':
      emotion_df.append(("fear", os.path.join(Tess,dir, wav)))
     elif info[2] == 'happy':
      emotion_df.append(("happy", os.path.join(Tess,dir, wav)))
     elif info[2] == 'neutral':
      emotion_df.append(("neutral", os.path.join(Tess,dir, wav)))
     elif info[2] == 'ps':
      emotion_df.append(("pleasant_surprised", os.path.join(Tess,dir, wav)))
     else:
      emotion_df.append(("unknown", os.path.join(Tess,dir, wav)))

df = pd.DataFrame(emotion_df)
df.rename(columns={1:"Path", 0:"Emotion"}, inplace=True)
print("Head of dataset:\n",df.head())
print("Shape of dataset:\n",df.shape)
print("Description of dataset:\n",df.describe())
plt.figure(figsize=(12,8))
sns.barplot(x=df['Emotion'].value_counts().index,y=df['Emotion'].value_counts().values)
plt.title("Number of cases in each emotion")
plt.show()
def create_waveplot(df, sr, e):
 plt.figure(figsize=(12, 8))
 plt.title(f'Waveplot for audio with {e} emotion', size=15)
 librosa.display.waveshow(df, sr=sr)
 plt.show()
def emo(emotion, df):
 path = np.array(df['Path'][df['Emotion'] == emotion])[1]
 df, sampling_rate = librosa.load(path)
 return df,sampling_rate,emotion,path

emotion='angry'
e1,e2,e3,path=emo(emotion,df)
create_waveplot(e1,e2,e3)
#playsound(path)
Audio(filename=path, autoplay=True)
emotion='fear'
e1,e2,e3,path=emo(emotion,df)
create_waveplot(e1,e2,e3)
#playsound(path)
Audio(filename=path, autoplay=True)
emotion='happy'
e1,e2,e3,path=emo(emotion,df)
create_waveplot(e1,e2,e3)
#playsound(path)
Audio(filename=path, autoplay=True)
emotion='sad'
e1,e2,e3,path=emo(emotion,df)
create_waveplot(e1,e2,e3)
#playsound(path)
Audio(filename=path, autoplay=True)
emotion='disgust'
e1,e2,e3,path=emo(emotion,df)
create_waveplot(e1,e2,e3)
#playsound(path)
Audio(filename=path, autoplay=True)
emotion='neutral'
e1,e2,e3,path=emo(emotion,df)
create_waveplot(e1,e2,e3)
#playsound(path)
Audio(filename=path, autoplay=True)
emotion='pleasant_surprised'
e1,e2,e3,path=emo(emotion,df)
create_waveplot(e1,e2,e3)
#playsound(path)
Audio(filename=path, autoplay=True)

df['Emotion'] = df['Emotion'].map({'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'neutral':4, 'pleasant_surprised':5, 'sad':6})
def extract_mfcc(filename):
    y,sr=librosa.load(filename,duration=3,offset=0.5)
    mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc
x_mfcc=df['Path'].apply(lambda x:extract_mfcc(x))
X=[x for x in x_mfcc]
X=np.array(X)
y=to_categorical(df['Emotion'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
model=Sequential([
    LSTM(123,return_sequences=False,input_shape=(40,1)),
    Dense(64,activation='relu'),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dropout(0.2),
    Dense(7,activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
history=model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=20,batch_size=25,shuffle=True)
acc= (model.evaluate(X_test,y_test)[1]*100)
print("Accuracy of model: ",round(acc,2),"%")
