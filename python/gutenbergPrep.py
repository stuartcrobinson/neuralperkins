import importlib
import os

import h

importlib.reload(h)

dir = '/Users/stuart.robinson/Gutenberg/txt'
files = [f for f in os.listdir(dir) if f.endswith('.txt')]

# files = [f for f in os.listdir(dir) if f.endswith('.txt')]

booksToAvoid = [f for f in files if ('poem' in f.lower() or 'poet' in f.lower() or 'play' in f.lower())]

# William Wymark Jacobs___For Better or Worse, Ship's Company, Part 10
authorsToAvoid = set([title.split('___')[0] for title in booksToAvoid])

files = [f for f in files if f.split('___')[0] not in authorsToAvoid]
#
# del booksToAvoid
# del authorsToAvoid

# hopefully we've removed all authors of plays and poems

# now for each file

# read file to string

# convert to lowercase + capschars

# get set of all characters, convert to list, and create index map

# convert file to list of indices

# for next files

# get set of all characters, and get list of unique characters (characters not seen already.  append this list to existing unique characters list and update index map

seglen = 100

# text = []
chars = []

m_char_index = {}
m_index_char = {}

indices = []

found = False

for file in files:
    if "Henry David Thoreau___A Week on the Concord and Merrimack Rivers" in file:
        found = True
    if not found:
        continue
    try:
        print(file)
        #
        file_text = h.getFileAsCharsArWCapsChar(dir + '/' + file)
        # print('len(file_text)', len(file_text))
        file_chars = set(file_text)
        # print('len(file_chars)', len(file_chars))
        #
        difference = sorted(file_chars.difference(list(set(chars))))
        # print('len(difference)', len(difference))
        #
        chars += difference
        # print(chars)
        #
        m_char_index, \
        m_index_char = h.getCharMaps(chars)
        #
        file_indices = [m_char_index[c] for c in file_text]
        #
        indices += file_indices
        #
        print(len(chars), len(indices))
    except:
        pass


#next? decode passages - see if it worked.  decode the last bit.
#if yes, save as numpy arrays.  indices, seglen, and chars

#next - try saving indices as int8 instead of int32.... oh shit no there's 299 indices now .... wtf.  shit.


# ugh.  now.  determine counts


import collections

indices_counts = collections.Counter(indices)

from operator import itemgetter

d = {"aa": 3, "bb": 4, "cc": 2, "dd": 1}



m_char_index, \
m_index_char = h.getCharMaps(chars)

tuples = []

for key, value in sorted(indices_counts.items(), key = itemgetter(1), reverse = True):
    # tuples += [(key, value)]
    print(m_index_char[key], value)

# tuples = tuples[0:int(len(tuples)/3)]
tuplesUnknown = tuples[int(len(tuples)/3):len(tuples)] #to make UNKNOWN

indicesUnknown = [i for (i, count) in tuplesUnknown]
unknownCharIndex = len(chars)

# m_char_index, \
# m_index_char = h.getCharMaps(chars)

m_char_index.update({h.getUnknownChar():unknownCharIndex})
m_index_char.update({unknownCharIndex:h.getUnknownChar()})




for unknownIndex in indicesUnknown:
    del m_index_char[unknownIndex]

new_chars = [c for (i, c) in m_index_char.items()]


new_m_char_index, \
new_m_index_char = h.getCharMaps(new_chars)

m_oldIndex_newIndex = {}

for c in new_chars:
    ni = new_m_char_index[c]
    oi = m_char_index[c]
    m_oldIndex_newIndex[oi] = ni

for ui in indicesUnknown:
    m_oldIndex_newIndex[ui] = new_m_char_index[h.getUnknownChar()]


for j, i in enumerate(indices):
    # print(j, i)
    indices[j] = m_oldIndex_newIndex[i]

# 7:31:19

import numpy as np

npIndices = np.zeros(len(indices), dtype='uint8')

for j, i in enumerate(indices):
    npIndices[j] = i


#7:45:34

import sys
sys.getsizeof(9)

'''
left with:
[m_index_char[i] for (i, count) in tuples]
[' ', 'e', 't', 'a', 'o', 'n', 'i', 'h', 's', 'r', 'd', 'l', 'ᚙ', 'u', 'm', '\n', 'c', 'w', 'f', 'g', 'y', ',', 'p', 'b', '.', 'v', 'k', '"', '-', "'", ';', 'x', 'j', 'q', '?', '!', '_', 'z', ':', '1', ')', '(', '0', '2', '3', '8', '5', '4', '’', '6', '7', '[', ']', '9', '“', '”', '*', '—', 'é', '=', '|', 'æ', '{', '}', '/', '&', '`', '‘', '\xa0', '#', 'ñ', 'á', 'è', '$', '£', 'â', '+', 'à', '§', 'ê', '‐', 'ô', '°', 'α', 'ν', 'ü', 'ö', 'ο', 'ε', 'τ', '\x97', 'ι', '·', 'ç', 'í', 'ä', 'ë', 'œ', 'ú', 'û', 'σ']
'''
# print('hi', end='', flush=True)

for i in range(0, len(npIndices)):
    print(new_m_index_char[npIndices[i]], end='', flush=True)

#now ... no, start with indices to replace with UNKNOWN char

#now hurry and save npIndices b4 i fuck something up

np.save(h.getRoot() + '/npIndices.npy', npIndices)
np.save(h.getRoot() + '/gbChars.npy', new_chars)

#
