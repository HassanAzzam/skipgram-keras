import numpy as np
import keras.preprocessing.text as keras_text
import keras.preprocessing.sequence as sequence
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
import sklearn.preprocessing as sklp
from keras import backend as K
import model
import zipfile
import tensorflow as tf
import collections

data_path = 'text8.zip'
vocab_size = 1000
features_dim = 300
context_size = 3

def main():
    data, count, dictionary, reverse_dictionary = build_dataset()
    target_input, context_input, labels = format_data(data)
    skipgram = model.SkipgramModel(vocab_size, features_dim)
    skipgram.main_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print 'model training: progress'
    skipgram.main_model.fit([target_input, context_input], labels, verbose=2)
    skipgram.main_model.save_weights('skipgram_weights.h5')

def build_dataset():
    words = read_data()
    n_words = vocab_size
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def format_data(data):
    sampling_table = sequence.make_sampling_table(vocab_size)
    couples, labels = sequence.skipgrams(data, vocab_size, window_size=context_size, sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")
    labels = np.array(labels, dtype="int32")
    return word_target, word_context, labels

def read_data():
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(data_path) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

main()
