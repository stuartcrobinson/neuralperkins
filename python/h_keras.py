
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.optimizers import RMSprop

def baseline_model(numUniqueChars, segmentLen):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(segmentLen, numUniqueChars)))
    model.add(LSTM(512))
    model.add(LSTM(512))
    model.add(Dense(numUniqueChars))
    model.add(Activation('softmax'))
    # optimizer = RMSprop(lr=0.01)
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
