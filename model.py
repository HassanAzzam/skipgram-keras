import numpy as np
import keras.preprocessing.text as keras_text
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Embedding, dot, Reshape
import sklearn.preprocessing as sklp
from keras import backend as K
import sys

class SkipgramModel:
    def __init__(self, vocab_size, features_dim):
        vocab_size = vocab_size
        features_dim = features_dim
        
        target_input = Input(shape=(1,), name='target_input')
        context_input = Input(shape=(1,), name='context_input')
        embedding_layer = Embedding(vocab_size, features_dim, input_length=1, name='embedding_layer')

        #encode target word
        target_encoded = embedding_layer(target_input)
        target_encoded = Reshape((features_dim, 1))(target_encoded)

        #encode context word    
        context_encoded = embedding_layer(context_input)
        context_encoded = Reshape((features_dim, 1))(context_encoded)
        
        #dot product two words
        dot_product = dot([target_encoded, context_encoded], axes=1)
        dot_product = Reshape((1,))(dot_product)

        #normalize
        output_layer = Dense(1, activation='sigmoid', name='output_layer')(dot_product)
        
        self.main_model = Model(inputs=[target_input, context_input], outputs=[output_layer])

        similarity = dot([target_encoded, context_encoded], axes=0, normalize=True)
        self.validation_model = Model(inputs=[target_input, context_input], outputs=[similarity])

