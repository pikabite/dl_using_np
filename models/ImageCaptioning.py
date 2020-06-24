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

class ImageCaptioning :

    embed_size = 1665
    total_t = 39

    def __init__ (self, istraining=True) :

        self.istraining = True
        self.sce_on_ts = []
        for t in range(self.total_t) :
            self.sce_on_ts.append(SoftmaxCrossEntropy())

        sc = 16 # start chennel
        # input shape = 64
        self.feature_extractor1 = [
            Conv2d(3, sc),
            Relu(),
            Conv2d(sc, sc),
            Relu(),
            AvgPool2d(2),
            Conv2d(sc, sc*2),
            Relu(),
            AvgPool2d(2),
            Conv2d(sc*2, sc*2),
            Relu(),
            AvgPool2d(2),
        ]
        self.feature_extractor2 = [
            Conv2d(sc*2, sc*4),
            Relu(),
            Concat(),
            AvgPool2d(2),
            Flatten(),
            LinearLayer(1536, 256),
            Tanh()
        ]
        self.text_feature_generator = [
            Rnn(300, 256),
        ]

        self.addlayer = Add()
        self.merger = [
            LinearLayer(256, 256),
            Relu(),
            LinearLayer(256, 1665)
        ]
        self.final_softmax = Softmax()

    def forward (self, x_img, y_text_embed) :

        output = np.zeros((y_text_embed.shape[0], y_text_embed.shape[1], self.embed_size))
        # Extract Image features
        image_feature = x_img
        for l in self.feature_extractor1 :
            image_feature = l.forward(image_feature)
            # print(image_feature.shape)
        skip_connection = image_feature
        for i, ll in enumerate(self.feature_extractor2) :
            tmp_input = image_feature if i != 2 else [image_feature, skip_connection]
            image_feature = ll.forward(tmp_input)

        # Extract Text features
        tfg = self.text_feature_generator[0]

        if self.istraining and y_text_embed is not None :
            # Through time
            for t in range(y_text_embed.shape[1]) :
                word_feature, hidden_state = tfg.forward(y_text_embed[:, i, :], np.zeros((y_text_embed.shape[0], 256)))
                merged_feature = self.addlayer.forward([image_feature, word_feature])
                for lll in self.merger :
                    merged_feature = lll.forward(merged_feature)
                    # merged_feature = self.final_softmax.forward(merged_feature)
                output[:, i, :] = merged_feature

        return output

    def calculate_loss (self, y_hat, y_text_onehot) :

        total_loss = []
        for t in range(y_text_onehot.shape[1]) :
            sce_on_t = self.sce_on_ts[t].forward(y_hat[:, t, :], y_text_onehot[:, t, :])
            total_loss.append(sce_on_t)
        return np.mean(np.array(total_loss).T)

    def backward (self, loss, y_text_onehot, lr=0.001) :

        to_word_f_on_t = []
        for i, sce in enumerate(self.sce_on_ts) :
            ii = len(self.sce_on_ts) - 1 - i
            dsce = sce.backward(y_text_onehot[:, ii, :], lr)
            dmerge_l = dsce
            for mer_l in self.merger[::-1] :
                dmerge_l = mer_l.backward(dmerge_l, lr)
            to_img_f, to_word_f = self.addlayer.backward(dmerge_l, lr)
            # print(to_img_f.shape)
            # print(to_word_f.shape)

            to_word_f_on_t.append(to_word_f)

            dfe2 = to_img_f
            for fe2 in self.feature_extractor2[::-1] :
                # print(fe2)
                # print(dfe2.shape)
                if fe2.__class__ == Concat : 
                    dfe2, dfe2_skip = fe2.backward(dfe2, lr)
                else :
                    dfe2 = fe2.backward(dfe2, lr)
            dfe1 = dfe2
            for fe1 in self.feature_extractor1[::-1] :
                dfe1 = fe1.backward(dfe1, lr)
            dfe_skip = dfe2_skip
            for fe1 in self.feature_extractor1[::-1] :
                dfe_skip = fe1.backward(dfe_skip, lr)

        # print(np.array(to_word_f_on_t).shape)
        self.text_feature_generator[0].backward(to_word_f_on_t, lr)




# tmp_input_img = np.random.random((3, 64, 64, 3))
# tmp_input_w_emb = np.random.random((3, 39, 300))
# tmp_input_w_label = np.random.randint(0, 1665, (3, 39))
# tmp_input_w_onehot = np.eye(1665)[tmp_input_w_label]
# tmptmp = ImageCaptioning()

# y_hat = tmptmp.forward(tmp_input_img, tmp_input_w_emb)
# total_loss = tmptmp.calculate_loss(y_hat, tmp_input_w_onehot)
# tmptmp.backward(total_loss, tmp_input_w_onehot)

# %%

