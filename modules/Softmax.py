#%%

import numpy as np


# %%

class Softmax :
    
    def __init__ (self) :
        self.tapes = {
            "softmax_out" : None
        }

    def softmax (self, input_data) :
        e = np.exp(input_data)
        eps = 1e-9
        return (e)/(np.sum(e, axis=1, keepdims=True)+eps)

    def forward (self, input_data) :
        softmax_out = self.softmax(input_data=input_data)
        self.tapes["softmax_out"] = softmax_out
        return softmax_out
    
    def backward (self, lr=0.001) :
        return self.tapes["softmax_out"] * (1 - self.tapes["softmax_out"])
