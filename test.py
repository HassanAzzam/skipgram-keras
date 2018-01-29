import numpy as np
import keras.preprocessing.text as keras_text
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
import sklearn.preprocessing as sklp
from keras import backend as K
import sys
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
    skipgram = model.SkipgramModel(vocab_size, features_dim)
    skipgram.main_model.load_weights('skipgram_weights.h5')
    while(1):
        print 'Enter word:'
        target = raw_input()
        target_idx = dictionary[target]
        top_k = 16  # number of nearest neighbors
        sim = get_simulation(skipgram.validation_model, target_idx)
        nearest = (-sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % target
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

def get_simulation(model, target_idx):
    sim = np.zeros((vocab_size,))
    in_arr1 = np.zeros((1,))
    in_arr2 = np.zeros((1,))
    for i in range(vocab_size):
        in_arr1[0,] = target_idx
        in_arr2[0,] = i
        out = model.predict_on_batch([in_arr1, in_arr2])
        sim[i] = out
    return sim
    
def build_dataset():
    words = read_data()
    print words[:40]
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

def read_data():
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(data_path) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

main()
