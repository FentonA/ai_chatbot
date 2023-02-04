import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pickle
import random
import nltk

from nltk.stem import WordNetLemmatizer
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout

import matplotlib.pyplot as plt
from matplotplib import rcParams


nltk.download('omw-1.4')
with open(r'intents.json') as data:
    intents = json.loads(data.read())
    
words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['pattern']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        classes.append(intent['tag'])
        documents.append((w, intent['tag']))
        
        
lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!']
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words, open('words.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words - doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower())]
    for w in words: 
        if w in pattern_words:
            bag.append(1)
        else: bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
random.shuffle(training)
a = int(0.7*len(training))
training = np.array(training, dtype = 'object')
X_train = list(training[:a, 0])
Y_train = list(training[:a, 1])
X_val = list(training[a, 0])
Y_val = list(training[a:, 1])

## Neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(Y_train[0]), activation='softmax'))

#Performing cost function, gradient descent and ....
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(X_train),
                 np.array(Y_train),
                 epochs=200,
                 batch_size=5, validation_data=(X_val, Y_val),
                 verbose=1
                )
model.save('trained_model.h5', hist)

#Plotting the data
plt.rcParams["figure.figsize"] = (12,8)
N = np.arange(0,200)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, hist.history["loss"], label="train_loss")
plt.plot(N, hist.history["val_loss"], label="val_loss")
plt.plot(N, hist.history['accuracy'], label="accuracy")
plt.plot(N, hist.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()