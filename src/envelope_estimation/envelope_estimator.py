import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, add
from keras.callbacks import EarlyStopping, ModelCheckpoint


DEFAULT_MODEL="../models/envelope_estimator/BLSTM_fourlang_60_60_augmented_dropout_v2.h5"


class EnvelopeEstimator:
    """
    BLSTM model for syllable envelope estimation.
    
    The model takes sequences of 24 MFCCs as input.
    The output of the BLSTM network is the activation of the output node for each
    row of MFFCs presented to the network in the input sequence (syllable envelope).

    Remark: This model is not yet trainable as we do not have training data, hence the 
    default model made by Okko Räsänen is always used.
    
    Attributes
    ----------
    model : Keras model
        Keras object containing the parameters of the model.
        
    Methods
    -------
    summary()
        Print a summary of the model.
    initialize_BLSTM_model(X_shape)
        Initialize a new untrained BLSTM model given an input shape.
    load_model(model_file)
        Load model from a file.
    train(X_train, y_train, model_file)
        Trains the model given the input 24 MFCCs sequences and their respective
        targeted output syllable envelopes.
    predict(X)
        Predict the syllable envelopes on a batch of MFCCs sequences.
    """
    
    def __init__(self):
        self.model = load_model(DEFAULT_MODEL)
        
    def summary(self):
        """
        Print a summary of the model.
        """
        
        self.model.summary()
        
    def initialize_BLSTM_model(self, X_shape):
        """
        Initialize a new untrained BLSTM model given an input shape.
        
        Parameters
        ----------
        X_shape : int tuple
            Shape of the input data to adapt the input of the model.
        """
        
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
        """
        Load model from a file.
        
        Parameters
        ----------
        model_file : str
            Path to model file.
        """
        
        try:
            self.model = load_model(model_file)
        except:
            print("Path to model is wrong.")
    
    def train(self, X_train, y_train, model_file="../models/trained_model.h5"):
        """
        Trains the model given the input 24 MFCCs sequences and their respective
        targeted output syllable envelopes.
        
        Parameters
        ----------
        X_train : ndarray
            3D array, batch of MFCCs sequences.
        y_train : ndarray
            2D array, targeted output syllable envelopes.
        """
        
        print("Training syllable envelope estimator model.")
        
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
        """
        Predict the syllable envelopes on a batch of MFCCs sequences.
        
        Parameters
        ----------
        X : ndarray
            3D array, batch of MFCCs sequences.
            
        Returns
        -------
        envelopes_batch : ndarray
            2D array, output syllable envelopes for each sequence.
        """
        
        print("Predicting envelopes batch.")
        
        if len(self.model.layers[0].output_shape) > 3:
            new_shape = [X.shape[0], X.shape[1], X.shape[2], 1]
            X = np.reshape(X, new_shape)
        
        envelopes_batch = self.model.predict_on_batch(X)
        if envelopes_batch.ndim > 2:
            envelopes_batch = envelopes_batch[:,:,0]

        print("Envelopes batch predicted successfully.")
        return envelopes_batch
