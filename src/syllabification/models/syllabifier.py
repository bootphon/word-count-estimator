"""
TODO
"""

import numpy as np
from keras.models import load_model

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, LSTM, add


DEFAULT_MODEL="../models/BLSTM_fourlang_60_60_augmented_dropout_v2.h5"


class Syllabifier:
    
    def __init__(self):
        self.model = load_model(DEFAULT_MODEL)
        
    def summary(self):
        self.model.summary()
        
    def initialize_BLSTM_model(self, X_shape):
        sequence = Input(shape=X_shape)

        forwards1 = LSTM(units=60, return_sequences=True)(sequence)
        backwards1 = LSTM(units=60, return_sequences=True,
                          go_backwards=True)(sequence)
        merged1 = add([forwards1, backwards1])

        forwards2 = LSTM(units=60, return_sequences=True)(merged1)
        backwards2 = LSTM(units=60, return_sequences=True,
                          go_backwards=True)(merged1)
        merged2 = add([forwards2, backwards2])

        outputvar = Dense(units=1, activation='sigmoid')(merged2)

        model = Model(outputs=outputvar, inputs=sequence)
        model.compile(loss='binary_crossentropy', 
                      optimizer='rmsprop',
                      metrics=['mean_squared_error'])
        
        self.model = model
        print("BLSTM model initialized successfully.")
        
    def load_model_from_file(self, model_file):
        try:
            self.model = load_model(model_file)
            print("Model loaded successfully.")
        except:
            print("Path to model is wrong.")
    
    def train(self, X_train, y_train):
        # TODO
        print("Train")
        
    def predict(self, X):
        if len(self.model.layers[0].output_shape) > 3:
            new_shape = [X.shape[0], X.shape[1], X.shape[2], 1]
            X = np.reshape(X, new_shape)
        
        envelopes = self.model.predict_on_batch(X)
        if envelopes.ndim > 2:
            envelopes = envelopes[:,:,0]
        
        print("Prediction finished successfully.")
        return envelopes
