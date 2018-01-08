import importlib
import numpy as np
import h_keras
import h
importlib.reload(h)
importlib.reload(h_keras)

root = h.getRoot()

text_str = "Alice was not a bit hurt, and she jumped up on to her feet in a moment. "

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

onehotArs = h.to_categorical(indices, num_classes=len(chars), dtype='bool')

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
    print(h.getCharHtml(m_index_char[nextCharI], pNextCharI, pmin, pmax, m_index_char[pmax_index]))

# html = h.getColoredHtmlText(output)
#
# print(html)
#
# with open("Output.html", "w") as text_file:
#     text_file.write(html)
