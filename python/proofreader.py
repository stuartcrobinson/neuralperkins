import importlib
import numpy as np
import h_keras
import h
importlib.reload(h)
importlib.reload(h_keras)

root = h.getRoot()


seglen = np.load(root + '/seglen.npy').item()
chars = np.load(root + '/chars.npy')


model = h_keras.baseline_model(len(chars), seglen)

model.load_weights(h.getWeightsFile())

m_char_index, \
m_index_char = h.getCharMaps(chars)


# text_str = "Alice was not a bit hurt, and she jumped up on to her feet in a moment. "

def getStrFromX(x):
    chars = []
    for onehot in x[0]:
        chars.append(m_index_char[np.argmax(onehot)])
    return ''.join(chars)

def myPad(x, seglen, numChars, spaceIndex):
    if len(x) < seglen:
        need = seglen - len(x)
        # print("need: ", need)
        # print(len(x))
        # print(x)
        zeros = np.zeros((need, numChars), dtype='bool')
        for row in zeros:
            row[spaceIndex] = 1
        x = np.concatenate((zeros, x))
        return x
    elif (len(x) > seglen):
        return x[-1*seglen:]
    else:
        return x

import scipy.stats as stats


def proofread(str):
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
        x = myPad(x, seglen, len(chars), spaceIndex)
        x = x.reshape((1, x.shape[0], x.shape[1]))
        # print(getStrFromX(x))
        preds = model.predict(x, verbose=0)[0]
        pNextCharI = preds[nextCharI]
        percentile = stats.percentileofscore(preds, pNextCharI) / 100.0
        pmin_ = np.amin(preds)
        pmax_index = np.argmax(preds)
        pmax_ = preds[pmax_index]
        output.append((m_index_char[nextCharI], pNextCharI, pmin_, pmax_, m_index_char[pmax_index]))
        print(m_index_char[nextCharI], "{:0.3f} {:.2e} {:.2e} {:.2e}".format(percentile, pNextCharI, pmin_, pmax_), h.getCharHtml(m_index_char[nextCharI], percentile, pNextCharI, pmin_, pmax_, m_index_char[pmax_index]))

# proofread("Now thad you’re heRe I cin remember things better.")



# proofread("I remember when.")


# proofread("I remember when.")
# run("Alice was not a bi. hurt, and she jumped up on to her feet in a moment.")
# run("CHAPTER I. Down the Rabbit-Hole")
# run("I would like a a cup of tea.")


