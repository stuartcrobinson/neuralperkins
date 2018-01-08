'''
take substrate text
convert to padded list of char arrays of length seglen

build numpy bool matrices x and y like for training

for each segment and next_char 1-hot vector arrays and vector,
plug the vector array into LSTM, get probaility of next_char as listed in output/predictions

add this tuple to a list of results:  char and probability.

'''

import numpy as np
import h


root = h.get_root()

text_str = "hello here are the “words”."



seglen = np.load(root + '/seglen.npy').item()
chars = np.load(root + '/chars.npy')

m_char_index, \
m_index_char = h.get_char_maps(chars)



indices = []

for char in text_str:
    if char.isupper():
        indices.append(m_char_index[h.chapsChar()])
        indices.append(m_char_index[char.lower()])
    else:
        indices.append(m_char_index[char])


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence


segments = []
next_indices = []
for i in range(0, len(indices)):
    segments.append(indices[min(i-seglen, 0) : i])
    print(segments[i])
    next_indices.append(indices[i])

print('nb sequences:', len(segments))


# xIndices = sequence.pad_sequences(segments, seglen)

# sequence.pad_sequences([['c', 'f', 'g']], 10)   # doesn't work

print('Vectorization...')
x = np.zeros((len(segments), seglen, len(chars)), dtype=np.bool)
y = np.zeros((len(segments), len(chars)), dtype=np.bool)
for i, segment in enumerate(segments):
    padded = sequence.pad_sequences([segment], seglen)[0]
    print(padded)
    for t, char_index in enumerate(padded):
        x[i, t, char_index] = 1
    print(x[i])
    y[i, next_indices[i]] = 1

#okay x and y are ready

model = h.baseline_model(len(chars), seglen)


output = [] # char_actual, p, pmin, pmax, char_pmax

for i, segment_one_hot_arrays in x:
    x_pred = segment_one_hot_arrays
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = np.argmax(y[i])
    next_index_probability = preds[next_index]
    pmin = np.amin(preds)
    pmax_index = np.argmax(preds)
    pmax = preds[pmax_index]
    output.append((m_index_char[next_index], next_index_probability, pmin, pmax, m_index_char[pmax_index]))


just_output_chars = [c for (c, p, pmin) in output]

html = h.get_colored_html_text(output)

print(html)

with open("Output.html", "w") as text_file:
    text_file.write(html)






