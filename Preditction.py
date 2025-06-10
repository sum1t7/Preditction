import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

# Load and preprocess text
path = '/kaggle/input/textdata/1661-0.txt'
text = open(path, encoding='utf-8').read().lower()
print('corpus length:', len(text))

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

unique_words = sorted(list(set(words)))
unique_word_index = {w: i for i, w in enumerate(unique_words)}
index_to_word = {i: w for w, i in unique_word_index.items()}

WORD_LENGTH = 5
prev_words = []
next_words = []

for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])

X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)

for i, each_words in enumerate(prev_words):
    for j, word in enumerate(each_words):
        X[i, j, unique_word_index[word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1

# Define model
model = Sequential()
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

# Save model and history
model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))

# Plotting
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Prediction helpers
def prepare_input(text):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    words = text.split()
    for t, word in enumerate(words):
        if word in unique_word_index:
            x[0, t, unique_word_index[word]] = 1
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10)  # Prevent log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completions(text, n=3):
    words = text.lower().split()
    if len(words) < WORD_LENGTH:
        print("Input text must be at least", WORD_LENGTH, "words.")
        return []
    input_seq = ' '.join(words[-WORD_LENGTH:])
    x = prepare_input(input_seq)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [index_to_word[idx] for idx in next_indices]

# Try it
quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]

for q in quotes:
    seq = ' '.join(q.lower().split()[:WORD_LENGTH])
    print(f"> {seq}")
    print(">>", predict_completions(seq, 5))
    print()
