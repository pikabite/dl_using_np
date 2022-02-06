#%%

import numpy as np

# %%

"""
Relu

simple relu activation function

TODO : add versions (leakyReLu, Relu6, etc...)

"""
class Relu :

    def __init__ (self) :
        self.tapes = {
            "x" : None
        }

    def forward (self, input_data) :
        self.tapes["x"] = input_data
        return np.maximum(input_data, 0)
    
    def backward (self, input_d, lr=0.001) :
        return input_d * ((self.tapes["x"] > 0) * 1)

