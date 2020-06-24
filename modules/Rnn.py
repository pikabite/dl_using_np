#%%

import numpy as np
from modules.Tanh import Tanh
from modules.Softmax import Softmax

# %%

"""

Rnn (output doesn't include softmax)

"""
class Rnn :
    
    def __init__ (self, input_channel, hidden_channel, activation="tanh") :

        self.m = input_channel
        self.n = hidden_channel
        
        self.u = np.random.normal(0.0, 0.01, (self.n, self.m))
        self.v = np.random.normal(0.0, 0.01, (self.n, self.n))
        self.w = np.random.normal(0.0, 0.01, (self.n, self.n))
        self.b = np.random.normal(0.0, 0.01, (self.n))
        self.activation = Tanh()
        self.softmax = Softmax()

        self.tapes = {
            "xt" : [],
            "ht-1" : [],
            "ht" : [],
        }


    def forward (self, input_data, input_state) :

        self.tapes = {
            "xt" : [],
            "ht-1" : [],
            "ht" : [],
        }

        self.tapes["xt"].append(input_data)
        self.tapes["ht-1"].append(input_state)
        new_state = self.activation.forward(self.u @ input_data.T + self.w @ input_state.T).T + self.b
        self.tapes["ht"].append(new_state)
        # output = self.softmax.forward(self.v @ new_state + self.b)
        output = (self.v @ new_state.T).T + self.b
        # output = (self.v @ new_state.T).T

        return output, new_state

    # backward input_d should indlude derivative of softmax
    def backward (self, input_d, lr=0.001) :

        # input_d must have t sequence of output loss
        du = np.zeros_like(self.u)
        dw = np.zeros_like(self.w)
        dv = np.zeros_like(self.v)
        db = np.zeros_like(self.b)

        for t in range(len(self.tapes["xt"]))[::-1] :
            dv += self.tapes["ht"][t].T @ input_d[t]
            dsigma = (self.v @ input_d[t].T) * (1 - np.square(self.tapes["ht"][t])).T

            for tt in range(max(0, t-5), t)[::-1] :
                # dw += dsigma @ input_d[tt]
                # du += dsigma @ input_d[tt]
                dw += dsigma @ self.tapes["ht-1"][tt]
                du += dsigma @ self.tapes["xt"][tt]
                dsigma = self.w @ dsigma * (1 - np.square(self.tapes["ht"][tt])).T

        self.update_uvwb(du, dw, dv, db, lr)

    def update_uvwb (self, du, dw, dv, db, lr=0.001) :

        self.u -= du*lr
        self.w -= dw*lr
        self.v -= dv*lr
        self.b -= db*lr


# def bptt(self, x, y):
#     T = len(y)
#     # Perform forward propagation
#     o, s = self.forward_propagation(x)
#     # We accumulate the gradients in these variables
#     dLdU = np.zeros(self.U.shape)
#     dLdV = np.zeros(self.V.shape)
#     dLdW = np.zeros(self.W.shape)
#     delta_o = o
#     delta_o[np.arange(len(y)), y] -= 1.
#     # For each output backwards...
#     for t in np.arange(T)[::-1]:
#         dLdV += np.outer(delta_o[t], s[t].T)
#         # Initial delta calculation: dL/dz
#         delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
#         # Backpropagation through time (for at most self.bptt_truncate steps)
#         for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
#             # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
#             # Add to gradients at each previous step
#             dLdW += np.outer(delta_t, s[bptt_step-1])
#             dLdU[:,x[bptt_step]] += delta_t
#             # Update delta for next step dL/dz at t-1
#             delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
#     return [dLdU, dLdV, dLdW]


# #%%


# for t in range(3)[::-1] :
#     print("t")
#     print(t)
#     for tt in range(max(0, t-3), t)[::-1] :
#         print("tt")
#         print(tt)
