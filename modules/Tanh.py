#%%

import numpy as np

# %%

class Tanh :

    def __init__ (self) :
        self.tapes = {
            "x" : None
        }
    
    def forward (self, input_data) :
        self.tapes["x"] = input_data
        return np.tanh(input_data)

    def backward (self, input_d, lr=0.001) :
        return (1 - np.square(np.tanh(self.tapes["x"]))) * input_d
