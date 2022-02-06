# %%

import numpy as np


# %%

"""

ADD Layer
Simple Summation of given tensors

TODO : Add axial addition
"""
class Add :

    def __init__ (self) :
        self.tapes = {
            "original_dims" : []
        }

    def forward (self, input_datas) :
        self.tapes["original_dims"] = []
        for dim in input_datas :
            self.tapes["original_dims"].append(dim)
        return np.sum(input_datas, axis=0)

    def backward (self, input_d, lr=0.001) :
        return [np.copy(input_d) for i in range(len(self.tapes["original_dims"]))]
