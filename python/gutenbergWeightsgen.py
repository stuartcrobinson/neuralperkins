import importlib
import random

import numpy as np

import h
import h_keras

importlib.reload(h)
importlib.reload(h_keras)

# len() is O(1)
'''
scp -i "eastKeyPairPem.pem" repos/neuralperkinsdata.zip ec2-user@ec2-34-224-37-168.compute-1.amazonaws.com:~

'''

root = h.getRoot()

indices = np.load(root + '/gbIndices.npy')
indices = indices[0:int(len(indices)/100)]
chars = np.load(root + '/gbChars.npy').item()

m_char_index, \
m_index_char = h.getCharMaps(chars)

# cut the text in semi-redundant sequences of maxlen characters
seglen = 40

np.save(root + '/gbSeglen.npy', seglen)

numSegments = len(indices) - seglen

# 1 2 3 4 5 6 7 8 9
# 1 2 3

print('Vectorization...')
x = np.zeros((numSegments, seglen, len(chars)), dtype=np.bool)
y = np.zeros((numSegments, len(chars)), dtype=np.bool)
for j in range(0, numSegments):
    next_i = indices[j + seglen]
    segment = indices[j:j + seglen]
    for t, i in enumerate(segment):
        x[j, t, i] = 1
    y[j, next_i] = 1

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


# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# filepath="weights-alice3hidden512.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

for iteration in range(1, 10000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y, batch_size=500, epochs=1)  # , callbacks=callbacks_list
    model.save_weights(weights_file, overwrite=True)
    count += 1
    # if count % 5 > 0:
    #     continue
    start_index = random.randint(0, len(text) - seglen - 1)
    #
    seed = text[start_index: start_index + seglen]
    # test = [m_index_char[i] for i in generated_indices]
    print('----- Generating with seed: "' + ''.join(seed) + '"')
    #
    print('using sample(): ')
    displayGenerate(model, seglen, m_index_char, m_char_index, seed, 150, sample)
    print('\nusing highest prob: ')
    displayGenerate(model, seglen, m_index_char, m_char_index, seed, 150, np.argmax)
    h_keras.proofread(m_char_index, m_index_char, seglen, chars, model, "I remember when you wexe a young child.  Things were were were simpler then.")



    # #
    # seed = [c for c in h.getCapsChar() + "alice sat down by the pond, but she knew she couldn’t stay long."]
    # seed = [c for c in h.getCapsChar() + "then the words don’t fit"]
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

# nohup python3 weightsgen.py > out.log 2>&1 &
