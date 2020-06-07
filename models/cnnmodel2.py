# %%

import numpy as np


# %%

class CNNmodel :

    def __init__(self):
        
        self.parameters = {
            "conv1_w" : np.random.normal(0.0, 0.05, (3, 3, 1, 4)),
            "conv1_b" : np.random.normal(0.0, 0.05, (4)),
            "conv2_w" : np.random.normal(0.0, 0.05, (3, 3, 4, 8)),
            "conv2_b" : np.random.normal(0.0, 0.05, (8)),
            "conv3_w" : np.random.normal(0.0, 0.05, (3, 3, 8, 16)),
            "conv3_b" : np.random.normal(0.0, 0.05, (16)),
            "conv4_w" : np.random.normal(0.0, 0.05, (3, 3, 16, 32)),
            "conv4_b" : np.random.normal(0.0, 0.05, (32)),
            "layer1_w" : np.random.normal(0.0, 0.05, (128, 10)),
            "layer1_b" : np.random.normal(0.0, 0.05, (10)),
        }
        self.momentum_velocities = {
        }
        self.forward_tape = {}

        self.momentum = 0.9

        self.lr = 0.01


    def im2col (self, input_data, filter_h, filter_w, stride=1, pad=0):
        """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
        
        Parameters
        ----------
        input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
        filter_h : 필터의 높이
        filter_w : 필터의 너비
        stride : 스트라이드
        pad : 패딩
        
        Returns
        -------
        col : 2차원 배열
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col


    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
        
        Parameters
        ----------
        col : 2차원 배열(입력 데이터)
        input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
        filter_h : 필터의 높이
        filter_w : 필터의 너비
        stride : 스트라이드
        pad : 패딩
        
        Returns
        -------
        img : 변환된 이미지들
        """
        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]


    def conv2d (self, input_data, kernel_size, stride, param, forward_tape, padding="SAME") :

        kernel = param
        # kernel = np.reshape(kernel, (kernel_size**2, kernel.shape[2], kernel.shape[3]))

        b, h, w, c = input_data.shape
        ph, pw, pic, poc = param.shape
        im_col = self.im2col(input_data.transpose((0, 3, 1, 2)), ph, pw, 1, 1)
        im_col_w = np.reshape(param, (poc, -1)).T

        output = (im_col @ im_col_w).reshape((b, h, w, poc))

        self.forward_tape[forward_tape] = input_data, im_col, im_col_w

        return output

    def avg_pool (self, input_data, stride) :

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


    def layer (self, input_data, param_name) :
        return input_data @ self.parameters[param_name+"_w"] + self.parameters[param_name+"_b"]

    def relu (self, input_data) :
        return np.maximum(input_data, 0)

    def softmax (self, input_data) :
        e = np.exp(input_data)
        eps = 1e-9
        return (e)/(np.sum(e, axis=1, keepdims=True)+eps)

    def cross_entropy (self, input_data, y) :
        y_onehot = np.eye(10)[y]
        ce = np.sum(- y_onehot * np.log(input_data), axis=1)
        return ce
    
    def dsce (self, loss, softmax, y) :
        y_onehot = np.eye(10)[y]
        return (softmax - y_onehot)/y.shape[0]

    def drelu (self, input_d, input_data) :
        return input_d * ((input_data > 0) * 1)

    def dlayer (self, input_d, input_data, target_w, target_b) :
        dw = input_data.T @ input_d / input_d.shape[0]
        db = np.sum(input_d, axis=0)
        self.update_wb(dw, db, target_w, target_b)
        return input_d @ dw.T

    def davg_pool (self, input_d, stride=2) :
        output = np.zeros((input_d.shape[0], input_d.shape[1]*stride, input_d.shape[2]*stride, input_d.shape[3]))
        for h in range(input_d.shape[1]) :
            for w in range(input_d.shape[2]) :
                output[:, h*stride:(h+1)*stride, w*stride:(w+1)*stride, :] = input_d[:, h:h+1, w:w+1, :]/4
        return output

    def dconv2d (self, input_d, input_datas, target_w, target_b) :

        input_data, im_col, im_col_w = input_datas
        param = self.parameters[target_w]

        ph, pw, pic, poc = param.shape

        input_d = input_d.reshape((-1, poc))

        db = np.sum(input_d, axis=0)
        dw = im_col.T @ input_d
        dw = dw.T.reshape((poc, pic, ph, pw))

        dx = self.col2im((input_d @ im_col_w.T), input_data.shape, ph, pw, 1, 1)

        return dx

    def update_wb (self, dw, db, target_w, target_b) :

        eps = 1e-7
        reg = 0.01

        # dw = np.clip(dw, -1., 1.)
        # db = np.clip(db, -1., 1.)

        # rms prop
        if target_w not in self.momentum_velocities :
            self.momentum_velocities[target_w] = 0
        if target_b not in self.momentum_velocities :
            self.momentum_velocities[target_b] = 0

        # regularization
        dw = dw + self.parameters[target_w] * reg
        db = db + self.parameters[target_b] * reg

        gw = self.momentum_velocities[target_w] * self.momentum + (1 - self.momentum) * np.square(dw)
        gb = self.momentum_velocities[target_b] * self.momentum + (1 - self.momentum) * np.square(db)
        self.momentum_velocities[target_w] = gw
        self.momentum_velocities[target_b] = gb

        vw = self.lr * dw / (np.sqrt(gw) + eps)
        vb = self.lr * db / (np.sqrt(gb) + eps)

        self.parameters[target_w] -= vw
        self.parameters[target_b] -= vb

    def forward (self, x) :
        if len(x.shape) < 4 :
            x = np.expand_dims(x, axis=-1)

        x = np.pad(
            x,
            (
                (0, 0),
                (2, 2),
                (2, 2),
                (0, 0),
            )
        )
        # print(x.shape)
        net = self.conv2d(x, 3, 1, self.parameters["conv1_w"], "input")
        net += self.parameters["conv1_b"]
        self.forward_tape["before_relu1"] = net.copy()
        net = self.relu(net)
        net = self.avg_pool(net, 2)
        # print(net.shape)

        net = self.conv2d(net, 3, 1, self.parameters["conv2_w"], "conv1")
        net += self.parameters["conv2_b"]
        self.forward_tape["before_relu2"] = net.copy()
        net = self.relu(net)
        net = self.avg_pool(net, 2)
        # print(net.shape)

        net = self.conv2d(net, 3, 1, self.parameters["conv3_w"], "conv2")
        net += self.parameters["conv3_b"]
        self.forward_tape["before_relu3"] = net.copy()
        net = self.relu(net)
        net = self.avg_pool(net, 2)
        # print(net.shape)

        net = self.conv2d(net, 3, 1, self.parameters["conv4_w"], "conv3")
        net += self.parameters["conv4_b"]
        self.forward_tape["before_relu4"] = net.copy()
        net = self.relu(net)
        net = self.avg_pool(net, 2)
        # print(net.shape)

        flatten = np.reshape(net, (net.shape[0], 128))
        # print(flatten.shape)
        self.forward_tape["flatten"] = flatten.copy()
        net = self.layer(flatten, "layer1")

        output = self.softmax(net)

        # print(output.shape)

        return output


    def backward (self, dL, softmax, y) :
        d1 = self.dsce(dL, softmax, y)
        d3 = self.dlayer(
            d1,
            self.forward_tape["flatten"],
            "layer1_w",
            "layer1_b"
        )
        reshaped_d3 = np.reshape(d3, (d3.shape[0], 2, 2, 32))
        d4 = self.davg_pool(reshaped_d3)
        d5 = self.drelu(
            d4,
            self.forward_tape["before_relu4"]
        )
        d6 = self.dconv2d(
            d5,
            self.forward_tape["conv3"],
            "conv4_w",
            "conv4_b"
        )
        d7 = self.davg_pool(d6)
        d8 = self.drelu(
            d7,
            self.forward_tape["before_relu3"]
        )
        d9 = self.dconv2d(
            d8,
            self.forward_tape["conv2"],
            "conv3_w",
            "conv3_b"
        )
        d10 = self.davg_pool(d9)
        d11 = self.drelu(
            d10,
            self.forward_tape["before_relu2"]
        )
        d12 = self.dconv2d(
            d11,
            self.forward_tape["conv1"],
            "conv2_w",
            "conv2_b"
        )
        d13 = self.davg_pool(d12)
        d14 = self.drelu(
            d13,
            self.forward_tape["before_relu1"]
        )
        d15 = self.dconv2d(
            d14,
            self.forward_tape["input"],
            "conv1_w",
            "conv1_b"
        )

    
    def calculate_loss (self, input_data, y) :
        return np.mean(self.cross_entropy(input_data, y))

    def preprocessing (self, image) :
        return image/256

    def train (self, x, y, lr=0.01) :

        self.lr = lr
        y_hat = self.forward(self.preprocessing(x))
        loss = self.calculate_loss(y_hat, y)
        self.backward(loss, y_hat, y)
        
        return loss

    def evaluate (self, x, y) :

        y_hat = self.forward(self.preprocessing(x))

        return np.sum(np.argmax(y_hat, axis=1) == y)*1/y_hat.shape[0], y_hat


model = CNNmodel()


from dataloader.mnist import MNIST_loader

dloader = MNIST_loader()

batch_size = 7

b_images, b_labels = dloader.get_batch(0, batch_size)

lr = 0.01
model.lr = lr

output = model.forward(b_images/256)
loss = model.calculate_loss(output, b_labels)

# print("asdasd")
# print(model.parameters["layer5_w"].shape)
# print(loss)

model.backward(loss, output, b_labels)


# %%
lr = 0.01

model = CNNmodel()
dloader = MNIST_loader()

epoch = 1

# model.forward_tape = {}
# model.momentum_velocities = {}

for e in range(epoch) : 
    for i in range(500) :

        ii = np.random.randint(0, 500)
        # ii = 50
        b_images, b_labels = dloader.get_batch(ii, batch_size)

        # print(model.parameters["conv1_w"][0][0])

        loss = model.train(b_images, b_labels, lr)

        print(i)
        print(loss)


# %%

i = np.random.randint(0, 1000)
b_images, b_labels = dloader.get_batch(i, 10)

output_acc, output = model.evaluate(b_images, b_labels)

print(output_acc)


# #%%


# %%
