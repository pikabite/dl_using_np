#%%


import numpy as np
from pathlib import Path
import pickle
import re
from dataloader.text import Tokenizer

from PIL import Image


# %%

class Caption_data :
    
    VOCAB_SIZE = 1665
    EMBEDDING_SIZE = 300
    MAX_TEXT_LEN = 39

    def __init__ (self, description_file, embedding_mat_file, image_folder) :

        with description_file.open("rb") as f:
            self.descriptions = pickle.load(f)
        with embedding_mat_file.open("rb") as f:
            self.embedding_mat = pickle.load(f)

        self.offset = 0
        self.image_root = image_folder
        self.imgnames = {}
        self.ids = []
        i = 0
        for v in image_folder.iterdir() :
            self.imgnames[v.stem] = self.image_root/v.name
            self.ids.append(v.stem)
            i += 1

        caption_list = []
        for el in self.descriptions.values():
            for ec in el:
                caption_list.append(ec)
        # print("The total caption present = {}".format(len(caption_list)))

        token = Tokenizer(num_words=self.VOCAB_SIZE)
        token.fit_on_texts(caption_list)
        self.token = token
        ix_to_word = token.index_word

        for k in list(ix_to_word):
            if k >= 1665:
                ix_to_word.pop(k, None)
        word_to_ix = dict()
        for k,v in ix_to_word.items():
            word_to_ix[v] = k
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        
        np.random.shuffle(self.ids)

    def batching_text (self, idid) :
        rand_seq_number = np.random.randint(0, len(self.descriptions[idid]))
        desc_seq = self.descriptions[idid][rand_seq_number].split(" ")

        txt_onehot = self.token.texts_to_matrix(desc_seq)
        tmp = np.zeros((self.MAX_TEXT_LEN, self.VOCAB_SIZE))
        # tmp[:, 0] = 1
        txt_onehot = np.concatenate([txt_onehot, tmp], axis=0)[:self.MAX_TEXT_LEN, :]
        txt_embed = self.embedding_mat[np.argmax(txt_onehot, axis=1)]
        # print(txt_onehot.shape)
        # print(txt_embed.shape)
        # print(self.word_to_ix["helocopter"])
        # print(self.embedding_mat[3])
        return txt_onehot, txt_embed

    def batching_image (self, idid) :
        img = Image.open(self.imgnames[idid], mode="r")
        img = img.resize((64, 64))
        img = np.asarray(img).copy().astype(np.float32)
        img = self.augmentation(img)
        return img
    
    def augmentation (self, img) :
        h, w, c = img.shape
        if np.random.random() < 0.5 :
            img = img[::-1, :, :]
        if np.random.random() < 0.5 :
            img = img[:, ::-1, :]
        if np.random.random() < 0.3 :
            noise = np.random.normal(0, 10, (h, w, 1))
            img += noise
        if np.random.random() < 0.3 :
            noise = np.random.randint(-30, 30, (h, w))
            zitter = np.zeros_like(img)
            zitter[:, :, np.random.randint(0, 3)] = noise
            img += zitter
        return img

    def batching (self, batch_size) :

        img_batch = []
        text_oh_batch = []
        text_em_batch = []
        offset = self.offset
        offsetend = min(len(self.ids) - 1, offset+batch_size)
        idids = self.ids[offset:offsetend]
        self.offset = offsetend

        for i in range(offsetend - offset) :
            img_batch.append(self.batching_image(idids[i]))
            txt_oh, txt_em = self.batching_text(idids[i])
            text_oh_batch.append(txt_oh)
            text_em_batch.append(txt_em)
            # print(self.batching_text(idids[i]).shape)

        if offsetend == len(self.ids) - 1 :
            self.offset = 0
            np.random.shuffle(self.ids)

        return np.array(img_batch), np.array(text_oh_batch), np.array(text_em_batch)

# desc_path = Path("datasets/descriptions.pkl")
# embed_path = Path("datasets/embedding_matrix.pkl")
# img_root = Path("datasets/Flicker8k_Dataset")
# tmptmp = Caption_data(desc_path, embed_path, img_root)
# tmpimg, tmptxt_oh, tmptxt_em = tmptmp.batching(10)

# print(np.array(tmptxt))

# %%
