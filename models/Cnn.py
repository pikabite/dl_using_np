#%%


import numpy as np
from modules.Conv2d import Conv2d
from modules.LinearLayer import LinearLayer
from modules.AvgPool2d import AvgPool2d
from modules.Flatten import Flatten
from modules.Relu import Relu
from modules.Rnn import Rnn
from modules.Softmax import Softmax
from modules.SoftmaxCrossEntropy import SoftmaxCrossEntropy
from modules.Concat import Concat
from modules.Tanh import Tanh
from modules.Add import Add


# %%

class Cnn :

    def __init__ (self, istraining=True) :

        self.istraining = True

        sc = 4 # start chennel
        # input shape = 64
        self.feature_extractor1 = [
            Conv2d(1, sc),
            Relu(),
            Conv2d(sc, sc),
            Relu(),
            AvgPool2d(2),
            Conv2d(sc, sc*2),
            Relu(),
            AvgPool2d(2),
            Conv2d(sc*2, sc*2),
            Relu(),
            # AvgPool2d(2),
        ]
        self.feature_extractor2 = [
            Conv2d(sc*2, sc*4),
            Relu(),
        ]
        self.feature_extractor3 = [
            Concat(),
            Conv2d(sc*6, sc*6),
            # AvgPool2d(2),
            Flatten(),
            # LinearLayer(1536, 10),
            LinearLayer(1176, 10),
            # Tanh(),
            # LinearLayer(256, 10)
        ]

        self.feature_extractor_tmp = [
            Conv2d(1, sc),
            Relu(),
            AvgPool2d(2),
            Conv2d(sc, sc),
            Relu(),
            AvgPool2d(2),
            Flatten(),
            LinearLayer(784, 10),

            # Flatten(),
            # LinearLayer(784, 392),
            # Relu(),
            # LinearLayer(392, 98),
            # Relu(),
            # LinearLayer(98, 10),
        ]

        self.final_sce = SoftmaxCrossEntropy()

    def forward (self, x_img) :

        # img_feature = x_img
        # for l in self.feature_extractor_tmp :
        #     img_feature = l.forward(img_feature)
        # return img_feature

        # Extract Image features
        image_feature = x_img
        for l in self.feature_extractor1 :
            image_feature = l.forward(image_feature)
        skip_connection = image_feature
        for i, ll in enumerate(self.feature_extractor2) :
            tmp_input = image_feature
            image_feature = ll.forward(tmp_input)
        for i, lll in enumerate(self.feature_extractor3) :
            # tmp_input = image_feature
            tmp_input = [image_feature, skip_connection] if lll.__class__ == Concat else image_feature
            # if len(tmp_input) > 2 : print(tmp_input.shape)
            image_feature = lll.forward(tmp_input)

        return image_feature

    def calculate_loss (self, y_hat, y) :
        sce = self.final_sce.forward(y_hat, y)
        return np.mean(sce)

    def backward (self, loss, y_onehot, lr=0.001) :

        # dsce = self.final_sce.backward(y_onehot, lr)
        # dfetmp = dsce
        # for l in self.feature_extractor_tmp[::-1] :
        #     dfetmp = l.backward(dfetmp, lr)

        # return dfetmp

        dsce = self.final_sce.backward(y_onehot, lr)
        dfe3 = dsce
        for fe3 in self.feature_extractor3[::-1] :
            if fe3.__class__ == Concat : 
                dfe3, dfe3_skip = fe3.backward(dfe3, lr)
            else :
                dfe3 = fe3.backward(dfe3, lr)
        
        dfe2 = dfe3
        for fe2 in self.feature_extractor2[::-1] :
            dfe2 = fe2.backward(dfe2, lr)

        dfe1 = dfe2
        for fe1 in self.feature_extractor1[::-1] :
            dfe1 = fe1.backward(dfe1, lr)

        dfe_skip = dfe3_skip
        for fe11 in self.feature_extractor1[::-1] :
            dfe_skip = fe11.backward(dfe_skip, lr)


# %%

tmptmp = Cnn()

from dataloader.mnist import MNIST_loader

mnist_loader = MNIST_loader()
mnist_loader.epoch_end()


# %%

for i in range(3000) :
    i = np.random.randint(0, 3000)
    # i = 10
    im, la = mnist_loader.get_batch(i, 20)
    y_hat = tmptmp.forward(im)
    total_loss = tmptmp.calculate_loss(y_hat, la)
    print(total_loss)
    tmptmp.backward(total_loss, la, lr=0.0001)

# %%
tmptim, tmptla = mnist_loader.get_test(20, 20)
tmptim = np.expand_dims(tmptim, axis=-1)/255
tmptout = tmptmp.forward(tmptim,)
tmptout.shape


# %%
acc = np.sum(((np.argmax(tmptout, axis=-1) == tmptla)*1)/20)
acc
