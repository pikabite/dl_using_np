# %%

from pathlib import Path
import pickle
import numpy as np
import idx2numpy
import numpy as np
# import cv2

# %%


class MNIST_loader :

    class_size = 10

    def __init__(self, root_folder="./datasets/mnist/"):

        self.rf = Path(root_folder)

        self.train = {
            "images" : [],
            "labels" : [],
            "indexes" : []
        }
        self.test = {
            "images" : [],
            "labels" : [],
            "indexes" : []
        }

        self.train["images"], self.train["labels"] = self.load_mnist(self.rf, "train")
        self.train["indexes"] += [i for i in range(0, 60000)]

        self.test["images"], self.test["labels"] = self.load_mnist(self.rf, "test")
        self.test["indexes"] += [i for i in range(0, 10000)]

        self.epoch_end()

    def load_mnist(self, root_folder_path, data_type):
        if data_type == "train" :
            filename1 = str(root_folder_path / "train-images.idx3-ubyte")
            filename2 = str(root_folder_path / "train-labels.idx1-ubyte")
        else :
            filename1 = str(root_folder_path / "t10k-images.idx3-ubyte")
            filename2 = str(root_folder_path / "t10k-labels.idx1-ubyte")

        images = idx2numpy.convert_from_file(filename1)
        labels = idx2numpy.convert_from_file(filename2)

        return images, labels

    def get_batch (self, step, batch_size) :

        offset = step*batch_size

        if offset >= len(self.train["indexes"]) :
            return None, None

        b_images = self.train["images"][self.train["indexes"][offset:offset+batch_size]]
        b_labels = self.train["labels"][self.train["indexes"][offset:offset+batch_size]]

        # b_images = self.train["images"][offset:offset+batch_size]
        # b_labels = self.train["labels"][offset:offset+batch_size]

        # b_images = [cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR) for img in b_images]
        # b_images = [cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR) for img in b_images]

        b_images = np.expand_dims(b_images, axis=-1)/255
        b_labels = np.eye(self.class_size)[b_labels]

        return b_images, b_labels

    def get_test (self, step, batch_size) :

        offset = step*batch_size

        if offset >= len(self.test["indexes"]) :
            return None, None

        b_images = self.test["images"][offset:offset+batch_size]
        b_labels = self.test["labels"][offset:offset+batch_size]

        b_images = np.expand_dims(b_images, axis=-1)/255
        b_labels = np.eye(self.class_size)[b_labels]
        
        return b_images, b_labels

    def epoch_end (self) :
        np.random.shuffle(self.train["indexes"])




# from PIL import Image

# mnist_loader = MNIST_loader()

# Image.fromarray(mnist_loader.train["images"][0])

# im, la = mnist_loader.get_batch(60, 99)


# %%

