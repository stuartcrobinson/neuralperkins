import importlib
import os
import random

import numpy as np
from keras.preprocessing import sequence

import h
import h_keras

importlib.reload(h)
importlib.reload(h_keras)

'''
scp -i "eastKeyPairPem.pem" repos/neuralperkinsdata.zip ec2-user@ec2-34-224-37-168.compute-1.amazonaws.com:~
    # 1 2 3 4 5 6 7 8 9
    # 1 2 3'''

root = h.getRoot()

chars = np.load(root + '/gbChars.npy')
seglen = 50
np.save(root + '/gbSeglen.npy', seglen)

m_char_index, \
m_index_char = h.getCharMaps(chars)

weights_file = h.getGbWeightsFile()
#
model = h_keras.baseline_model(len(chars), seglen)

if os.path.isfile(weights_file):
    model.load_weights(weights_file)

# iter = 200
for iter in range(0, 999):
    #
    indices = np.load(root + '/gbIndices.npy')
    textlen = int(len(indices) / 1000)
    #
    indices = indices[textlen * iter: textlen * (1 + iter)]
    #
    numSegments = len(indices) - seglen
    #
    print('Vectorization...')
    x = np.zeros((numSegments, seglen, len(chars)), dtype=np.bool)
    y = np.zeros((numSegments, len(chars)), dtype=np.bool)
    for j in range(0, numSegments):
        next_i = indices[j + seglen]
        segment = indices[j:j + seglen]
        for t, i in enumerate(segment):
            x[j, t, i] = 1
        y[j, next_i] = 1
    #
    model.fit(x, y, batch_size=600, epochs=1)  # , callbacks=callbacks_list
    model.save_weights(weights_file, overwrite=True)
    #
    start_index = random.randint(0, len(indices) - seglen - 1)
    #
    seed = [m_index_char[i] for i in indices[start_index: start_index + seglen]]
    # test = [m_index_char[i] for i in generated_indices]
    print('----- Generating with seed: "' + ''.join(seed) + '"')
    #
    print('\nusing sample: ')
    h_keras.displayGenerate(model, seglen, m_index_char, m_char_index, seed, 300, h_keras.sample, temperature=2)
    print('\nusing highest prob: ')
    h_keras.displayGenerate(model, seglen, m_index_char, m_char_index, seed, 300, np.argmax)
    h_keras.proofread(m_char_index, m_index_char, seglen, chars, model, "I remember when you wexe a young child.  Things were were were simpler then.")
    #
    del indices

# #
# sys.getsizeof(x)
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

# nohup python3 gutenbergWeightsgen.py > out.log 2>&1 &
