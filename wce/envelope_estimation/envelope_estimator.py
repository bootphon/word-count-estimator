import os, sys
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, add
from keras.callbacks import EarlyStopping, ModelCheckpoint


class EnvelopeEstimator:
    """
    BLSTM model for syllable envelope estimation.

    The model takes a matrix of MFCCs as input.
    The output of the BLSTM network is the activation of the output node for each
    frame of MFFCs presented to the network in the input sequence. This output
    represents the syllable probability. Once the whole signal is processed it
    gives the syllable probability envelope.

    Remark: This model is not yet trainable as we do not have training data,
    hence the default model made by Okko Räsänen is always used.

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
        Train the model given the input MFCCs frames and their respective
        target output syllable envelopes.
    predict(X)
        Predict the syllable envelopes on a batch of MFCCs frames.
    """

    def summary(self):
        """
        Print a summary of the model.
        """

        if not self.model:
            sys.exit("Initialize/Load envelope estimator model first.")
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
        except IOError:
            print("Path to envelope estimation model is wrong.")

    def train(self, X_train, y_train, model_file, itermediate_model_file):
        """
        Trains the model given the input MFCCs frames and their respective
        targeted output syllable envelopes.

        Parameters
        ----------
        X_train : ndarray
            3D array, batch of MFCCs frames.
        y_train : ndarray
            2D array, targeted output syllable envelopes of the frames.
        """
        
        if not self.model:
            sys.exit("Initialize/Load envelope estimator model first.")

        new_shape = [y_train.shape[0], y_train.shape[1], 1]
        y_train = np.reshape(y_train, new_shape)

        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                      patience=15, verbose=0, mode='auto')
        try:
            checkPoint = ModelCheckpoint(intermediate_model_file,
                                         monitor='val_loss')
        except IOError:
            print("Invalid intermediate envelope estimator model path.")

        self.model.fit(X_train, y_train, validation_data=(X_train, y_train),
                       shuffle=True, epochs=15000, batch_size=250,
                       callbacks=[earlyStopping, checkPoint],
                       validation_split=0.1)

        try:
            self.model.save(model_file)
        except IOError:
            print("Invalid envelope estimator model path.")

    def predict(self, X):
        """
        Predict the syllable envelopes on a batch of MFCCs frames.

        Parameters
        ----------
        X : ndarray
            3D array, batch of MFCCs frames.

        Returns
        -------
        envelope_batch : ndarray
            2D array, output syllable envelopes of the frames.
        """

        if not self.model:
            sys.exit("Initialize/Load envelope estimator model first.")

        if len(self.model.layers[0].output_shape) > 3:
            new_shape = [X.shape[0], X.shape[1], X.shape[2], 1]
            X = np.reshape(X, new_shape)

        envelope_batch = self.model.predict_on_batch(X)
        if envelope_batch.ndim > 2:
            envelope_batch = envelope_batch[:, :, 0]

        return envelope_batch

