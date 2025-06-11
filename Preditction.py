import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, Dropout, Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
from collections import Counter

path = './1661-0.txt'
text = open(path, encoding='utf-8').read().lower()
print('corpus length:', len(text))

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

word_counts = Counter(words)
min_word_count = 2
words = [word for word in words if word_counts[word] >= min_word_count]

unique_words = sorted(list(set(words)))
unique_word_index = {w: i for i, w in enumerate(unique_words)}
index_to_word = {i: w for w, i in unique_word_index.items()}

print(f'Total words: {len(words)}')
print(f'Unique words: {len(unique_words)}')

SEQUENCE_LENGTH = 10  
VOCAB_SIZE = len(unique_words)

prev_words = []
next_words = []
for i in range(len(words) - SEQUENCE_LENGTH):
    prev_words.append(words[i:i + SEQUENCE_LENGTH])
    next_words.append(words[i + SEQUENCE_LENGTH])

print(f'Number of sequences: {len(prev_words)}')

X = np.zeros((len(prev_words), SEQUENCE_LENGTH), dtype=np.int32)
y = np.zeros((len(next_words),), dtype=np.int32)

for i, word_sequence in enumerate(prev_words):
    for j, word in enumerate(word_sequence):
        X[i, j] = unique_word_index[word]
    y[i] = unique_word_index[next_words[i]]

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

model = Sequential([
    Embedding(VOCAB_SIZE, 128, input_length=SEQUENCE_LENGTH),
    LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    LSTM(256, dropout=0.3, recurrent_dropout=0.3),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(VOCAB_SIZE, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['accuracy'])

print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.0001
)

history = model.fit(
    X, y,
    validation_split=0.1,
    batch_size=128,
    epochs=20,  
    callbacks=[early_stopping, reduce_lr],
    verbose=1
).history

model.save('improved_keras_model.h5')
pickle.dump(history, open("improved_history.p", "wb"))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

def prepare_input(text):
    words = text.lower().split()
    if len(words) < SEQUENCE_LENGTH:
        words = [''] * (SEQUENCE_LENGTH - len(words)) + words
    
    words = words[-SEQUENCE_LENGTH:]
    
    x = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int32)
    for i, word in enumerate(words):
        if word in unique_word_index:
            x[0, i] = unique_word_index[word]
    return x

def sample_with_temperature(preds, temperature=0.8, top_k=10):
    preds = np.asarray(preds).astype('float64')
    
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    top_k_indices = np.argpartition(preds, -top_k)[-top_k:]
    top_k_probs = preds[top_k_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    
    choice = np.random.choice(top_k_indices, p=top_k_probs)
    return choice

def predict_completions(text, n=5, temperature=0.8):
    words = text.lower().split()
    if len(words) < SEQUENCE_LENGTH:
        print(f"Input text should be at least {SEQUENCE_LENGTH} words for best results.")
    
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    
    top_indices = np.argsort(preds)[-n:][::-1]
    predictions = []
    
    for idx in top_indices:
        word = index_to_word[idx]
        confidence = preds[idx]
        predictions.append((word, confidence))
    
    return predictions

def generate_text(seed_text, num_words=20, temperature=0.8):
    """Generate a sequence of words"""
    result = seed_text.lower().split()
    
    for _ in range(num_words):
        input_text = ' '.join(result[-SEQUENCE_LENGTH:])
        x = prepare_input(input_text)
        preds = model.predict(x, verbose=0)[0]
        
        next_idx = sample_with_temperature(preds, temperature)
        next_word = index_to_word[next_idx]
        result.append(next_word)
    
    return ' '.join(result)

quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]

print("\n" + "="*60)
print("TESTING MODEL")
print("="*60)

for i, quote in enumerate(quotes):
    print(f"\nQuote {i+1}: {quote}")
    seed = ' '.join(quote.lower().split()[:SEQUENCE_LENGTH])
    print(f"Seed: {seed}")
    
    predictions = predict_completions(seed, 5)
    print("Next word predictions:")
    for word, conf in predictions:
        print(f"  {word}: {conf:.4f}")
    
    generated = generate_text(seed, 10)
    print(f"Generated text: {generated}")
    print("-" * 40)