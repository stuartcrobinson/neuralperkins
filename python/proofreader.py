'''
take substrate text
convert to padded list of char arrays of length seglen

build numpy bool matrices x and y like for training

for each segment and next_char 1-hot vector arrays and vector,
plug the vector array into LSTM, get probaility of next_char as listed in output/predictions

add this tuple to a list of results:  char and probability.

'''

import importlib

import numpy as np

import h_keras
import h

importlib.reload(h)
importlib.reload(h_keras)

root = h.getRoot()

text_str = "Alice was not"  # a bit hurt, and she jumped up on to her feet in a moment. "

seglen = np.load(root + '/seglen.npy').item()
chars = np.load(root + '/chars.npy')

m_char_index, \
m_index_char = h.getCharMaps(chars)

from keras.utils import to_categorical

indices = []

for char in text_str:
    if char.isupper():
        indices.append(m_char_index[h.getCapsChar()])
        indices.append(m_char_index[char.lower()])
    else:
        indices.append(m_char_index[char])


model = h_keras.baseline_model(len(chars), seglen)

model.load_weights(h.getWeightsFile())

onehotArs = to_categorical(indices)

output = []  # char_actual, p, pmin, pmax, char_pmax

for j in range(0, len(onehotArs)):
    nextCharI = np.argmax(onehotArs[j])
    x = onehotArs[min(j - seglen, 0): j]
    x = h.myPad(x, seglen, len(chars))
    x = x.reshape((1, x.shape[0], x.shape[1]))
    preds = model.predict(x, verbose=0)[0]
    pNextCharI = preds[nextCharI]
    pmin = np.amin(preds)
    pmax_index = np.argmax(preds)
    pmax = preds[pmax_index]
    output.append((m_index_char[nextCharI], pNextCharI, pmin, pmax, m_index_char[pmax_index]))

html = h.getColoredHtmlText(output)

print(html)

with open("Output.html", "w") as text_file:
    text_file.write(html)



#
# import h_keras
#
# model = h_keras.baseline_model(len(chars), seglen)
#
# model.load_weights(h.getWeightsFile())
#
# output = []  # char_actual, p, pmin, pmax, char_pmax
#
# for i, segment_one_hot_arrays in enumerate(x):
#     x_pred = segment_one_hot_arrays
#     shape = x_pred.shape
#     x_pred = x_pred.reshape((1, shape[0], shape[1]))
#     # print(x_pred)
#     preds = model.predict(x_pred, verbose=0)[0]
#     next_index = np.argmax(y[i])
#     next_index_probability = preds[next_index]
#     pmin = np.amin(preds)
#     pmax_index = np.argmax(preds)
#     pmax = preds[pmax_index]
#     output.append((m_index_char[next_index], next_index_probability, pmin, pmax, m_index_char[pmax_index]))
# #
# for i, segment_one_hot_arrays in enumerate(x):
#     x_pred = segment_one_hot_arrays
#     shape = x_pred.shape
#     x_pred = x_pred.reshape((1, shape[0], shape[1]))
#     # print(x_pred)
#     preds = model.predict(x_pred, verbose=0)[0]
#     next_index = np.argmax(y[i])
#     next_index_probability = preds[next_index]
#     pmin = np.amin(preds)
#     pmax_index = np.argmax(preds)
#     pmax = preds[pmax_index]
#     output.append((m_index_char[next_index], next_index_probability, pmin, pmax, m_index_char[pmax_index]))


#
# from keras.preprocessing import sequence
#
# segments = []
# next_indices = []
# for i in range(0, len(indices)):
#     segments.append(indices[min(i - seglen, 0): i])
#     # print(segments[i])
#     next_indices.append(indices[i])
#
# print('nb sequences:', len(segments))
#
# # xIndices = sequence.pad_sequences(segments, seglen)
#
# # sequence.pad_sequences([['c', 'f', 'g']], 10)   # doesn't work
#
# print('Vectorization...')
# x = np.zeros((len(segments), seglen, len(chars)), dtype=np.bool)
# y = np.zeros((len(segments), len(chars)), dtype=np.bool)
# for i, segment in enumerate(segments):
#     padded = sequence.pad_sequences([segment], seglen)[0]
#     # print(padded)
#     for t, char_index in enumerate(padded):
#         x[i, t, char_index] = 1
#     # print(x[i])
#     y[i, next_indices[i]] = 1

# okay x and y are ready
