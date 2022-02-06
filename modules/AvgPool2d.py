#%%

import numpy as np


# %%

"""

AvgPool2D
Average Pooling on 2 dimensional feature


"""
class AvgPool2d :

    def __init__ (self, stride=2) :
        self.stride = stride

    def forward (self, input_data) :

        stride = self.stride
        output = np.zeros(
            (
                input_data.shape[0], 
                int(input_data.shape[1]/stride + input_data.shape[1]%stride), 
                int(input_data.shape[2]/stride + input_data.shape[2]%stride), 
                input_data.shape[3]
            )
        )

        for h in range(0, input_data.shape[1], stride) :
            for w in range(0, input_data.shape[2], stride) :
                target = input_data[:, h:h+2, w:w+2, :]
                output[:, int(h/stride), int(w/stride), :] = np.mean(target, axis=(1, 2))

        return output

    def backward (self, input_d, lr=0.001) :

        stride = self.stride
        output = np.zeros((input_d.shape[0], input_d.shape[1]*stride, input_d.shape[2]*stride, input_d.shape[3]))
        for h in range(input_d.shape[1]) :
            for w in range(input_d.shape[2]) :
                output[:, h*stride:(h+1)*stride, w*stride:(w+1)*stride, :] = input_d[:, h:h+1, w:w+1, :]/4
        return output
