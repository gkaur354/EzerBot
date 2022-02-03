#Description: This model classifies phrases as 1 (indicative of emotional distress -- for example: suicidal
#ideation, depression, anxiety, etc) or 0 (other)

#Import libraries
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
import pickle

from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam

#load the data 
data = pd.read_csv("/Users/gurnirmalkaur/Desktop/pathBASEF/BasefDatasetUpdate.csv")
print(data.head())

#Count of 0 and 1 in labeled dataset
counts = data['intention'].value_counts()
print(counts)

#List of stopwords
stop_words = stopwords.words('english')

#Remove I, me, my and myself from stopwords because they provide important information 
last_index = len(stop_words) - 1
stopWords = stop_words[4:last_index]

with open('stopwords_pickle', 'wb') as f:
    pickle.dump(stopWords,f)

#Data preprocessing:
#Remove punctuation and numbers from text
def only_letters(text):
    import re
    pattern = r'[0-9]'
    from string import punctuation
    final = ""
    for i in text:
        if i not in punctuation:
            new_word = re.sub(pattern, '', i)
            final += new_word
    return final

data['phrase'] = data['phrase'].apply(lambda x:only_letters(x))

#Convert to lower case
data['phrase'] = data['phrase'].apply(lambda x: x.lower())

#Remove stopwords from text
def remove_stopwords(text):
    phrase = nltk.word_tokenize(text)
    noStopwords = ""
    for i in phrase:
        if i not in stopWords:
            noStopwords += i
            noStopwords += " "
    return noStopwords

data['phrase'] = data['phrase'].apply(lambda x:remove_stopwords(x))

#Create corpus 
from nltk.tokenize import word_tokenize
def create_corpus(data):
    corpus = []
    for phrase in data['phrase']:
        tokenized_phrase = word_tokenize(phrase)
        words = []
        for word in tokenized_phrase:
            words.append(word)
        corpus.append(words)
    return corpus

corpus = create_corpus(data)

#Spit data into train and test 
X = data['phrase']
y = data['intention']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Covert to array
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

#Save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_sequences = tokenizer.texts_to_sequences(X_train)

#Pad train sequences 
maximum = 50
padded_train = pad_sequences(train_sequences, maxlen=maximum, truncating='post', padding='post')

#Pad test sequences
test_sequences = tokenizer.texts_to_sequences(X_test)
padded_test = pad_sequences(test_sequences, maxlen=maximum, truncating='post', padding='post')

word_index = tokenizer.word_index

#Create embedding dictionary
glove_vectors = dict()
with open("/Users/gurnirmalkaur/Desktop/pathBASEF/glove.6B.100d.txt", "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:])
        glove_vectors[word] = vectors

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 100))

for word, i in word_index.items():
    if i < num_words:
        emb_vec = glove_vectors.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec

#Create model 
model = Sequential()
model.add(Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),input_length=maximum,trainable=False))
model.add(LSTM(100, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=3e-4)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",mode="min",patience=10, restore_best_weights=True)

hist = model.fit(padded_train,Y_train, epochs=80, validation_data=(padded_test, Y_test),verbose=1,callbacks=earlystopping)
model.save('chatbotmodel.h5', hist)

#89%

