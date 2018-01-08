import random
import sys

import numpy as np

import h
import h_keras

import importlib
importlib.reload(h)
importlib.reload(h_keras)

# len() is O(1)

root = h.getRoot()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


text = []

with open('texts/aliceInWonderland.txt') as f:
    for line in f:
        for char in line:
            if char.isupper():
                text.append(h.getCapsChar())
                text.append(char.lower())
            else:
                text.append(char)

print(len(text))

chars = sorted(list(set(text)))

np.save(root + '/chars.npy', chars)

print(len(chars))

m_char_index, \
m_index_char = h.getCharMaps(chars)

# cut the text in semi-redundant sequences of maxlen characters
seglen = 40

np.save(root + '/seglen.npy', seglen)

segments = []
next_chars = []
for i in range(0, len(text) - seglen):
    segments.append(text[i: i + seglen])
    next_chars.append(text[i + seglen])

print('nb sequences:', len(segments))

print('Vectorization...')
x = np.zeros((len(segments), seglen, len(chars)), dtype=np.bool)
y = np.zeros((len(segments), len(chars)), dtype=np.bool)
for i, sentence in enumerate(segments):
    for t, char in enumerate(sentence):
        x[i, t, m_char_index[char]] = 1
    y[i, m_char_index[next_chars[i]]] = 1

model = h_keras.baseline_model(len(chars), seglen)

weights_file = h.getWeightsFile()

import os

if os.path.isfile(weights_file):
    model.load_weights(weights_file)

from keras.preprocessing import sequence

# train the model, output generated text after each iteration
count = 0


def displayGenerate(model, seglen, m_index_char, m_char_index, seed, n, sample):
    generated_indices = [m_char_index[c] for c in seed]
    for i in range(n):
        x_pred = np.zeros((1, seglen, len(chars)))
        for t, char_i in enumerate(sequence.pad_sequences([generated_indices], seglen)[0]):
            x_pred[0, t, char_i] = 1.
        #
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds)
        #
        generated_indices += [next_index]
        #
        print(m_index_char[next_index], end='', flush=True)
    print()


for iteration in range(1, 10000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y, batch_size=500, epochs=1)
    model.save_weights(weights_file, overwrite=True)
    count += 1
    if count % 5 > 0:
        continue
    start_index = random.randint(0, len(text) - seglen - 1)
    #
    seed = text[start_index: start_index + seglen]
    # test = [m_index_char[i] for i in generated_indices]
    print('----- Generating with seed: "' + ''.join(seed) + '"')
    #
    print('using sample(): ')
    displayGenerate(model, seglen, m_index_char, m_char_index, seed, 400, sample)
    print('\nusing highest prob: ')
    displayGenerate(model, seglen, m_index_char, m_char_index, seed, 400, np.argmax)

    # #
    # seed = [c for c in h.getCapsChar() + "alic"]
    # generated_indices = [m_char_index[c] for c in seed]
    # print('----- Generating with seed: "' + ''.join(seed) + '"')
    # for i in range(2):
    #     x_pred = np.zeros((1, seglen, len(chars)))
    #     for t, char_i in enumerate(sequence.pad_sequences([generated_indices], seglen)[0]):
    #         x_pred[0, t, char_i] = 1.
    #     #
    #     preds = model.predict(x_pred, verbose=0)[0]
    #     next_index = np.argmax(preds)
    #     #
    #     generated_indices += [next_index]
    #     #
    #     print(m_index_char[next_index], end='')
    #     sys.stdout.flush()
    # print()

# nohup python3 weightsgen.py >/dev/null 2>&1 &
