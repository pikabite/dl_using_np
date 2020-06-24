#%%

import numpy as np


# %%

class SoftmaxCrossEntropy :
    def __init__ (self) :
        self.tapes = {
            "softmax_out" : None
        }

    def softmax (self, input_data) :
        # input_data = input_data - np.max(input_data)
        e = np.exp(input_data)
        eps = 1e-9
        return (e)/(np.sum(e, axis=-1, keepdims=True)+eps)

    def cross_entropy (self, input_data, y) :
        # y_onehot = np.eye(self.class_num)[y]
        ce = np.sum(- y * np.log(input_data), axis=1)
        return ce

    def forward (self, input_data, y) :
        softmax_out = self.softmax(input_data=input_data)
        self.tapes["softmax_out"] = softmax_out
        cross_entropy_out = self.cross_entropy(input_data=softmax_out, y=y)

        return cross_entropy_out

    def backward (self, y, lr=0.001) :
        return ((self.tapes["softmax_out"] - y)/y.shape[0])


# %%
