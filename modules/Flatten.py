#%%

import numpy as np

# %%

class Flatten :

    def __init__ (self) :
        self.tapes = {
            "x_shape" : None
        }

    def forward (self, input_data) :
        self.tapes["x_shape"] = input_data.shape
        return np.reshape(input_data, (input_data.shape[0], -1))

    def backward (self, input_d, lr=0.001) :
        return np.reshape(input_d, self.tapes["x_shape"])
