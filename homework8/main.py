import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
stop_words = stopwords.words('english')

#function which stop words and the words less than 2 characters
def remove_stop_words (text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
      result.append(token)
  return result

# load the dataset
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

df_true['is_fake'] = 0
df_fake['is_fake'] = 1
df_final = pd.concat([df_true, df_fake]).reset_index(drop = True)
df_final.drop(columns = ['date', 'subject', 'title'], inplace = True)
df_final['text'] = df_final['text'].apply(remove_stop_words)
df_final['text'] = df_final['text'].apply(lambda x: " ".join(x))

#splitting the Data
x_train, x_test, y_train, y_test = train_test_split(df_final.text, df_final.is_fake, test_size = 0.2)

#word tokernizing
max_words = 1000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(x_train)
train_sequence = tokenizer.texts_to_sequences(x_train)
test_sequence = tokenizer.texts_to_sequences(x_test)

#Add padding
pad_train = pad_sequences(train_sequence, maxlen = 500, padding = 'post', truncating = 'post' )
pad_test = pad_sequences(test_sequence, maxlen = 500, padding = 'post')

def LSTM_modle():
  model = Sequential()
  model.add(Embedding(max_words, output_dim = 64)),
  model.add(LSTM(64))
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(1, activation = 'sigmoid'))

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['acc'])
  
  return model

model = LSTM_modle()
y_train = np.asarray(y_train)

#model training
model.fit(pad_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)

#prediction
predicted_result = model.predict(pad_test)

#if predicted value > 0.5 its real else fake
prediction = []
for i in range (len(predicted_result)):
  if predicted_result[i].item() > 0.5:
    prediction.append(1)
  else:
    prediction.append(0)

#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(list(y_test), prediction)

print(accuracy)