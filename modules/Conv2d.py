#%%

import numpy as np

# %%

"""
Conv2d

Convolution operation layer.
include initialize weights, convolution, back propagation

for simple operation, applied im2col tricks in forward backward

TODO : implement multiple initializer, cuda code, etc...

"""
class Conv2d :

    def __init__ (self, input_channel, output_channel, kernel_size=3, strides=1, padding="SAME") :

        w_shape = (kernel_size, kernel_size, input_channel, output_channel)
        
        # Normal initializer
        # self.w = np.random.normal(0, np.sqrt(2/input_channel), w_shape)
        
        # Random uniform initializer
        # self.w = np.random.normal(0, 0.05, w_shape)
        
        # He normal initializer
        self.w = np.sqrt(1/input_channel) * np.random.randn(*w_shape)
        self.b = np.zeros((output_channel))
        self.tapes = {
            "x" : None,
            "img_col" : None,
            "w_col" : None
        }
        self.strides = strides

        self.m = 0.9
        self.vw = 0
        self.vb = 0
    
    # for bhwc
    def im2col (self, input_data, filter_h, filter_w, stride=1, pad=0):
        N, H, W, C = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input_data, [(0,0), (pad, pad), (pad, pad), (0,0)], 'constant')
        col = np.zeros((N, filter_h, filter_w, out_h, out_w, C))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, y, x, :, :, :] = img[:, y:y_max:stride, x:x_max:stride, :]

        col = col.transpose(0, 3, 4, 5, 1, 2).reshape(N*out_h*out_w, -1)
        return col


    def forward (self, input_data) :

        b, h, w, c = input_data.shape
        ph, pw, pic, poc = self.w.shape
        # padding same
        im_col = self.im2col(input_data, ph, pw, self.strides, 1)
        im_col_w = np.reshape(self.w, (poc, -1)).T

        output = (im_col @ im_col_w).reshape((b, h//self.strides, w//self.strides, poc))

        self.tapes["x"] = input_data
        self.tapes["img_col"] = im_col
        self.tapes["w_col"] = im_col_w

        return output

    def backward (self, input_d, lr=0.001) :

        ph, pw, pic, poc = self.w.shape
        im_col = self.tapes["img_col"]
        w_col = self.tapes["w_col"]

        # print(input_d.shape)
        # input_d = input_d.transpose(0, 3, 1, 2).reshape((-1, poc))
        # input_d = input_d.transpose(0, 2, 1, 3).reshape((-1, poc))
        
        # input_d_col = self.im2col(np.tile(input_d, reps=(1, 1, 1, pic)), ph, pw, self.strides, 1)
        input_d_col = self.im2col(input_d, ph, pw, self.strides, 1)
        input_d_flat = input_d.reshape((poc, -1)).T

        db = np.sum(input_d_flat, axis=0)
        # print("im_col.shape")
        # print(im_col.shape)
        # print("input_d_flat.shape")
        # print(input_d_flat.shape)
        dw = im_col.T @ input_d_flat
        dw = dw.reshape((ph, pw, pic, poc))

        # input_d_col = np.tile(input_d_col, (pic, pic))
        # w_col = w_col.reshape(ph*pw, pic)
        w_col = w_col.reshape(-1, pic)[::-1, :]
        # w_col = w_col[::-1, :]
        # print("input_d_col.shape")
        # print(input_d_col.shape)
        # print("w_col.shape")
        # print(w_col.shape)
        # print("self.w")
        # print(np.squeeze(self.w))
        # print("w_col")
        # print(w_col)

        dx = (input_d_col @ w_col).reshape(self.tapes["x"].shape)

        dw += 1e-14
        db += 1e-14
        dx += 1e-14

        self.update_wd(dw, db, lr)

        return dx

    def update_wd (self, dw, db, lr=0.001) :

        self.vw = self.vw * self.m - dw*lr
        self.vb = self.vb * self.m - db*lr

        self.w += self.vw
        self.b += self.vb

# %%

# x = np.array([
#     [
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#     ]
# ])
# x = np.expand_dims(x, axis=-1)
# x = np.array([
#     [
#         [[1, 1], [0, 0], [0, 0], [0, 0]],
#         [[0, 0], [1, 1], [0, 0], [0, 0]],
#         [[0, 0], [0, 0], [1, 1], [0, 0]],
#         [[0, 0], [0, 0], [0, 0], [1, 1]],
#     ],
#     [
#         [[1, 1], [0, 0], [0, 0], [0, 0]],
#         [[0, 0], [1, 1], [0, 0], [0, 0]],
#         [[0, 0], [0, 0], [1, 1], [0, 0]],
#         [[0, 0], [0, 0], [0, 0], [1, 1]],
#     ]
# ])
# print(x.shape)

# y = np.array([
#     [
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 0.5],
#     ],
#     [
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 0.5],
#     ]
# ])
# y = np.array([
#     [
#         [1, 0],
#         [0, 1],
#     ],
#     [
#         [1, 0],
#         [0, 1],
#     ]
# ])
# y = np.expand_dims(y, axis=-1)
# y = np.array([
#     [1, 0, 0, 1],
#     [1, 0, 0, 1],
# ])
# y = np.expand_dims(y, axis=-1)
# print(y.shape)

# from modules.Relu import Relu
# from modules.AvgPool2d import AvgPool2d
# from modules.Flatten import Flatten

# l = Conv2d(2, 7, 3, 1)
# r = Relu()
# av = AvgPool2d(2)
# l3 = Conv2d(7, 1, 3, 1)
# fl = Flatten()

# l.w = np.ones_like(l.w)
# l2.w = np.ones_like(l2.w)
# l.b = np.zeros((3))

# for i in range(200) :
#     yhat = fl.forward(l3.forward((av.forward(r.forward(l.forward(x))))))
#     # yhat = l.forward(x)
#     print("np.squeeze(yhat)")
#     print(np.squeeze(yhat))
#     dy = 2*(yhat - y)
#     dl = l.backward(r.backward(av.backward(l3.backward(fl.backward(dy), lr=0.01), lr=0.01), lr=0.01), lr=0.01)
#     # dl = l.backward(dy, lr=0.01)


# %%
