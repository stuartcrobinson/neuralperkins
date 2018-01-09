
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.optimizers import RMSprop

def baseline_model(numUniqueChars, segmentLen):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(segmentLen, numUniqueChars)))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(numUniqueChars))
    model.add(Activation('softmax'))
    # optimizer = RMSprop(lr=0.01)
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

import numpy as np
import h

import scipy.stats as stats


def proofread(m_char_index, m_index_char, seglen, chars, model, str):
    indices = []
    for char in str:
        if char.isupper():
            indices.append(m_char_index[h.getCapsChar()])
            indices.append(m_char_index[char.lower()])
        else:
            indices.append(m_char_index[char])
    onehotArs = h.to_categorical(indices, num_classes=len(chars), dtype='bool')
    output = []  # char_actual, p, pmin, pmax, char_pmax
    spaceIndex = m_char_index[' ']
    for j in range(0, len(onehotArs)):
        nextCharI = np.argmax(onehotArs[j])
        x = onehotArs[max(j - seglen, 0): j]
        x = h.myPad(x, seglen, len(chars), spaceIndex)
        x = x.reshape((1, x.shape[0], x.shape[1]))
        # print(getStrFromX(x, m_index_char))
        preds = model.predict(x, verbose=0)[0]
        pNextCharI = preds[nextCharI]
        percentile = stats.percentileofscore(preds, pNextCharI) / 100.0
        pmin_ = np.amin(preds)
        pmax_index = np.argmax(preds)
        pmax_ = preds[pmax_index]
        output.append((m_index_char[nextCharI], pNextCharI, pmin_, pmax_, m_index_char[pmax_index]))
        print(m_index_char[nextCharI], "{:0.3f} {:.2e} {:.2e} {:.2e}".format(percentile, pNextCharI, pmin_, pmax_), h.getCharHtml(m_index_char[nextCharI], percentile, pNextCharI, pmin_, pmax_, m_index_char[pmax_index]))
