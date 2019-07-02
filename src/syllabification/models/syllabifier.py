"""
TODO
"""

import numpy as np
from keras.models import load_model

class Syllabifier:
    
    def __init__(self, model_file):
        try:
            self.model = load_model(model_file)
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
        
        return envelopes
