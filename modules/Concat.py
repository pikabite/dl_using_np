#%%

import numpy as np


# %%
"""
Concat
Concatenate multipl tensors

"""
class Concat :
    
    def __init__ (self) :
        self.tapes = {
            "original_dims" : []
        }

    def forward (self, input_datas) :
        self.tapes["original_dims"] = []
        for dim in [layer.shape[-1] for layer in input_datas] :
            self.tapes["original_dims"].append(dim)
        return np.concatenate(input_datas, axis=-1)
    
    def backward (self, input_d, lr=0.001) :
        self.tapes["original_dims"]
        tmp = 0
        tmptmp = []
        for v in self.tapes["original_dims"] :
            tmptmp.append(v+tmp)
            tmp += v
        return np.split(input_d, tmptmp[:-1], axis=-1)
