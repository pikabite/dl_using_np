# %%

import numpy as np


# %%

class NN5model :
    
    def __init__(self):
        
        self.parameters = {
            "layer1_w" : np.random.normal(0, 0.1, (784, 784*2)),
            "layer1_b" : np.random.normal(0, 0.1, (784*2)),
            "layer2_w" : np.random.normal(0, 0.1, (784*2, 784*2)),
            "layer2_b" : np.random.normal(0, 0.1, (784*2)),
            "layer3_w" : np.random.normal(0, 0.1, (784*2, 784*2)),
            "layer3_b" : np.random.normal(0, 0.1, (784*2)),
            "layer4_w" : np.random.normal(0, 0.1, (784*2, 784*2)),
            "layer4_b" : np.random.normal(0, 0.1, (784*2)),
            "layer5_w" : np.random.normal(0, 0.1, (784*2, 10)),
            "layer5_b" : np.random.normal(0, 0.1, (10)),
        }
        self.forward_tape = {}

        self.lr = 0.01
    

    def layer (self, input_data, param_name) :
        return input_data @ self.parameters[param_name+"_w"] + self.parameters[param_name+"_b"]

    def relu (self, input_data) :
        return np.maximum(input_data, 0)

    def softmax (self, input_data) :
        e = np.exp(input_data)
        return e/np.sum(e, axis=1, keepdims=True)
    
    def cross_entropy (self, input_data, y) :

        y_onehot = np.eye(10)[y]
        ce = np.sum(- y_onehot * np.log(input_data), axis=1)

        return ce
    
    def dsce (self, loss, softmax, y) :
        y_onehot = np.eye(10)[y]
        return softmax - y_onehot

    def drelu (self, input_d, input_data) :
        return input_d * ((input_data > 0) * 1)

    def dlayer (self, input_d, input_data, target_w, target_b) :

        dw = input_data.T @ input_d / input_d.shape[0]
        db = np.sum(input_d, axis=0)

        self.update_wb(dw, db, target_w, target_b)

        return input_d @ dw.T / input_d.shape[0]
    
    def update_wb (self, dw, db, target_w, target_b) :
        self.parameters[target_w] -= dw * self.lr
        self.parameters[target_b] -= db * self.lr


    def forward (self, x) :

        # print(x.shape)

        flatten = np.reshape(x, (x.shape[0], -1))

        # print(flatten.shape)

        self.forward_tape["input"] = flatten.copy()
        net = self.layer(input_data=flatten, param_name="layer1")
        self.forward_tape["before_relu1"] = net.copy()
        net = self.relu(net)
        self.forward_tape["layer1"] = net.copy()
        net = self.layer(input_data=net, param_name="layer2")
        self.forward_tape["before_relu2"] = net.copy()
        net = self.relu(net)
        self.forward_tape["layer2"] = net.copy()
        net = self.layer(input_data=net, param_name="layer3")
        self.forward_tape["before_relu3"] = net.copy()
        net = self.relu(net)
        self.forward_tape["layer3"] = net.copy()
        net = self.layer(input_data=net, param_name="layer4")
        self.forward_tape["before_relu4"] = net.copy()
        net = self.relu(net)
        self.forward_tape["layer4"] = net.copy()
        net = self.layer(input_data=net, param_name="layer5")

        output = self.softmax(net)

        return output

    def backward (self, dL, softmax, y) :

        d1 = self.dsce(dL, softmax, y)
        d3 = self.dlayer(
            d1,
            self.forward_tape["layer4"],
            "layer5_w",
            "layer5_b"
        )
        d4 = self.drelu(
            d3,
            self.forward_tape["before_relu4"]
        )
        d5 = self.dlayer(
            d4,
            self.forward_tape["layer3"],
            "layer4_w",
            "layer4_b"
        )
        d6 = self.drelu(
            d5,
            self.forward_tape["before_relu4"]
        )
        d7 = self.dlayer(
            d6,
            self.forward_tape["layer2"],
            "layer3_w",
            "layer3_b"
        )
        d8 = self.drelu(
            d7,
            self.forward_tape["before_relu4"]
        )
        d9 = self.dlayer(
            d8,
            self.forward_tape["layer1"],
            "layer2_w",
            "layer2_b"
        )
        d10 = self.drelu(
            d9,
            self.forward_tape["before_relu4"]
        )
        d11 = self.dlayer(
            d10,
            self.forward_tape["input"],
            "layer1_w",
            "layer1_b"
        )
    
    def calculate_loss (self, input_data, y) :
        return np.mean(self.cross_entropy(input_data, y))

    def preprocessing (self, image) :
        return image/256

    def train (self, x, y, lr) :

        self.lr = lr
        y_hat = self.forward(self.preprocessing(x))
        loss = self.calculate_loss(y_hat, y)
        self.backward(loss, y_hat, y)
        
        return loss

    def evaluate (self, x, y) :

        y_hat = self.forward(self.preprocessing(x))

        return np.sum(np.argmax(y_hat, axis=1) == y)*1/y_hat.shape[0], y_hat



# model = NN5model()

# from dataloader.mnist import MNIST_loader

# dloader = MNIST_loader()
# b_images, b_labels = dloader.get_batch(0, 7)


# output = model.forward(b_images/256)
# loss = model.calculate_loss(output, b_labels)
# model.backward(loss, output, b_labels)

# print(loss)

# #%%

# for i in range(100) :

#     i = np.random.randint(0, 1000)
#     b_images, b_labels = dloader.get_batch(i, 10)

#     loss = model.train(b_images, b_labels, 0.001)

#     print(loss)

#     # print(model.parameters["layer5_w"][0, 0])

# # %%


# i = np.random.randint(0, 1000)
# b_images, b_labels = dloader.get_batch(i, 10)

# output = model.evaluate(b_images, b_labels)

# print(output)


# #%%

# model.lr = 0.0001
