#%%

import numpy as np

# %%

class LinearLayer :
    
    def __init__ (self, input_channel, output_channel) :

        w_shape = (input_channel, output_channel)
        # self.w = np.random.normal(0, np.sqrt(2/input_channel), w_shape)
        # self.w = np.random.normal(0, 0.05, w_shape)
        self.w = np.sqrt(1/input_channel) * np.random.randn(*w_shape)
        self.b = np.zeros((output_channel))
        self.tapes = {
            "x" : None
        }

        self.m = 0.9
        self.vw = 0
        self.vb = 0

    def forward (self, input_data) :
        self.tapes["x"] = input_data
        return (input_data @ self.w) + self.b
        # return input_data @ self.w

    def backward (self, input_d, lr=0.001) :
        input_data = self.tapes["x"]
        dw = (input_data.T @ input_d)
        db = np.sum(input_d, axis=0)
        # dx = input_d @ dw.T
        dx = input_d @ self.w.T

        self.update_wb(dw, db, lr)

        dw += 1e-14
        db += 1e-14
        dx += 1e-14

        return dx

    def update_wb (self, dw, db, lr=0.001) :

        self.vw = self.vw * self.m - dw*lr
        self.vb = self.vb * self.m - db*lr

        self.w += self.vw
        self.b += self.vb


# %%


# x = np.random.random((1, 3))
# # x = np.array([[0.1, 1, 0.1], [1, 0.1, 0.1]])
# print(x.shape)
# y = np.array([[0, 1, 0]])
# print(y.shape)

# from modules.Relu import Relu
# from modules.SoftmaxCrossEntropy import SoftmaxCrossEntropy

# l = LinearLayer(3, 3)
# r = Relu()
# l2 = LinearLayer(3, 3)
# r2 = Relu()
# l3 = LinearLayer(3, 3)
# sf = SoftmaxCrossEntropy()

# # l.w = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
# # l.b = np.zeros((3))

# for i in range(200) :
#     yhat = l3.forward(r2.forward(l2.forward(r.forward(l.forward(x)))))
#     # yhat = l.forward(x)
#     loss = sf.forward(yhat, y)
#     print(yhat)
#     print(loss)
#     # dy = yhat - y
#     dsce = sf.backward(y)
#     dl3 = l3.backward(dsce, lr=0.01)
#     dr2 = r2.backward(dl3, lr=0.01)
#     dl2 = l2.backward(dr2, lr=0.01)
#     dr = r.backward(dl2, lr=0.01)
#     dl = l.backward(dr, lr=0.01)



# %%
