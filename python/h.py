'''
helper functions
'''


def getCharMaps(chars):
    m_char_index = dict((c, i) for i, c in enumerate(chars))
    m_index_char = dict((i, c) for i, c in enumerate(chars))
    return m_char_index, m_index_char


import os


def getRoot():
    root = '../../neuralperkinsdata' # there should only be two dotdots here
    try:
        os.makedirs(root)
    except:
        pass
    return root


import numpy as np

def getCapsChar():
    chapsChar = 'ᚙ'
    return chapsChar

def getUnknownChar():
    unknownChar = 'ᚊ'
    return unknownChar

##
def getRgbString(r, g, b):
    return ''.join(['rgb(', str(int(r)), ',', str(int(g)), ',', str(int(b)), ')'])

def getWhiteYellowShade(p, pmax):
    pct = p / pmax
    r = 255
    g = 255
    b = 255 * pct
    print('white, ', str(pct), ', ', str(r), ', ', str(g),  ', ', str(b))
    return getRgbString(r, g, b)


def getYellowRedShade(p, pmean):
    pct = p / pmean
    r = 255
    g = 255 * pct
    b = 0
    print('yellow, ', str(pct), ', ', str(r), ', ', str(g),  ', ', str(b))
    return getRgbString(r, g, b)



def getYellowRedShade_singleSpectrum(p, pmin, pmax):
    pct = p / pmax
    r = 255
    g = 255 * pct
    b = 0
    # print('yellow, ', str(pct), ', ', str(r), ', ', str(g),  ', ', str(b))
    return getRgbString(r, g, b)


def getYellowRedShade_percentile(percentile):
    return getRgbString(255, 255*percentile, 0)


def getRgbColor(p, percentile, pmin, pmax):
    # pmean = np.mean([pmin, pmax])
    # if p > pmean:
    #     return getWhiteYellowShade(p, pmax)
    # else:
    #     return getYellowRedShade(p, pmean)
    return getYellowRedShade_percentile(percentile)


def htmlEncode(char):
    if char == ' ':
        return '&nbsp;'
    if char == '\n':
        return '<br/>'
    return char

def getColoredHtmlText(output):
    '''
<span style="background-color: rgb(255,255,200)">A</span
><span style="background-color: rgb(255,255,150)">A</span
><span style="background-color: rgb(255,255,100)">A</span
><span style="background-color: rgb(255,255,50)">A</span
><span style="background-color: rgb(255,255,25)">A</span
><span style="background-color: rgb(255,255,0)">A</span
><span style="background-color: rgb(255,230,0)">1</span
><span style="background-color: rgb(255,210,0)">2</span
    :param chars_and_probs:
    :return:
    best way build strings:
    http://xahlee.info/python/python_append_string_in_loop.html
    https://waymoot.org/home/python_string/
    '''

    strings = ['<span></span\n']  # so all lines can begin with '>'
    # need to make some characters html safe.  like ' '
    for (char_actual, p, percentile, pmin, pmax, char_pmax) in output:
        strings.append('><span style="background-color: ')
        strings.append(getRgbColor(p, percentile, pmin, pmax))
        strings.append('" title="')
        strings.append(htmlEncode(char_pmax))
        strings.append('">')
        strings.append(htmlEncode(char_actual))
        strings.append('</span\n')
        pass
    strings.append('><br/><br/>')
    return ''.join(strings)


def getCharHtml(char, percentile, pchar, pmin, pmax, charSuggest):
    strings = []
    strings.append('<span style="background-color: ')
    strings.append(getRgbColor(pchar, percentile, pmin, pmax))
    strings.append('" title="')
    strings.append(htmlEncode(charSuggest))
    strings.append('">')
    strings.append(htmlEncode(char))
    strings.append('</span>')
    return ''.join(strings)


def getWeightsFile():
    return getRoot() + '/weights3layer_2_notoverfitting'


def getGbWeightsFile():
    return getRoot() + '/weights9layerGutenberg'

# def getGbWeights9File():
#     return getRoot() + '/weights9layerGutenberg'


def to_categorical(ar, num_classes, dtype='bool'):
    matrix = np.zeros((len(ar), num_classes), dtype=dtype)
    for i, v in enumerate(ar):
        matrix[i][v] = 1
    return matrix

#    print(h.getCharHtlm(m_index_char[nextCharI], pNextCharI, pmin, pmax, m_index_char[pmax_index]))


def getStrFromX(x, m_index_char):
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


def getFileAsCharsArWCapsChar(file):
    text = []
    with open(file) as f:
        for line in f:
            for char in line:
                if char.isupper():
                    text.append(getCapsChar())
                    text.append(char.lower())
                else:
                    text.append(char)
    return text