"""
TODO
"""

import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, LSTM, add
from keras.callbacks import EarlyStopping, ModelCheckpoint


DEFAULT_MODEL="../models/envelope_estimator/BLSTM_fourlang_60_60_augmented_dropout_v2.h5"


class EnvelopeEstimator:
    
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
        
    def load_model(self, model_file):
        try:
            self.model = load_model(model_file)
            print("Model loaded successfully.")
        except:
            print("Path to model is wrong.")
    
    def train(self, X_train, y_train, model_file="../models/curr_model.h5"):
        new_shape = [y_train.shape[0], y_train.shape[1],1]
        y_train = np.reshape(y_train, new_shape)
        
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                      patience=15, verbose=0, mode='auto')
        checkPoint = ModelCheckpoint("../models/curr_intermed.h5",
                                     monitor='val_loss')
        
        self.model.fit(X_train, y_train, validation_data=(X_train, y_train),
                       shuffle=True, epochs=15000,batch_size=250,
                       callbacks=[earlyStopping, checkPoint],
                       validation_split=0.1)
        self.model.save(model_file)
        print("Training finished successfully.")
        print("Model saved at {}.".format(model_file))
        
    def predict(self, X):
        if len(self.model.layers[0].output_shape) > 3:
            new_shape = [X.shape[0], X.shape[1], X.shape[2], 1]
            X = np.reshape(X, new_shape)
        
        envelope_windows = self.model.predict_on_batch(X)
        if envelope_windows.ndim > 2:
            envelope_windows = envelope_windows[:,:,0]

        print("Envelopes windows predicted successfully.")
        return envelope_windows
