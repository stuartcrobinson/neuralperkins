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


def run(str):
    indices = []
    for char in str:
        if char.isupper():
            indices.append(m_char_index[h.getCapsChar()])
            indices.append(m_char_index[char.lower()])
        else:
            indices.append(m_char_index[char])
    onehotArs = h.to_categorical(indices, num_classes=len(chars), dtype='bool')
    output = []  # char_actual, p, pmin, pmax, char_pmax
    # html = h.getColoredHtmlText(output)
    #
    # print(html)
    #
    # with open("Output.html", "w") as text_file:
    #     text_file.write(html)
    spaceIndex = m_char_index[' ']
    count = 0
    for j in range(0, len(onehotArs)):
        count += 1
        if count > 100:
            break
        nextCharI = np.argmax(onehotArs[j])
        x = onehotArs[min(j - seglen, 0): j]
        x = h.myPad(x, seglen, len(chars), spaceIndex)
        x = x.reshape((1, x.shape[0], x.shape[1]))
        # print(getStrFromX(x))
        preds = model.predict(x, verbose=0)[0]
        pNextCharI = preds[nextCharI]
        pmin = np.amin(preds)
        pmax_index = np.argmax(preds)
        pmax = preds[pmax_index]
        output.append((m_index_char[nextCharI], pNextCharI, pmin, pmax, m_index_char[pmax_index]))
        print(m_index_char[nextCharI], h.getCharHtml(m_index_char[nextCharI], pNextCharI, pmin, pmax, m_index_char[pmax_index]))


run("Alice was not")# a bit hurt, and she jumped up on to her feet in a moment.")



