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



# proofread("Now thad you’re heRe I cin remember things better.")



# proofread("I remember when.")


# proofread("I remember when.")
# run("Alice was not a bi. hurt, and she jumped up on to her feet in a moment.")
# run("CHAPTER I. Down the Rabbit-Hole")
# run("I would like a a cup of tea.")


