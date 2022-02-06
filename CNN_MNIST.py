#%%

from models.ImageCaptioning import ImageCaptioning
from models.Cnn import Cnn
from dataloader.caption_data import Caption_data
from dataloader.mnist import MNIST_loader
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
import seaborn as sn
import pandas as pd

#%%

model = Cnn()

dataloader = MNIST_loader()
dataloader.epoch_end()

modelname="SIMPLE_CNN_MNIST"
logfile = "logs/log_"+modelname+".csv"
lossfig_save = "images/loss_graph_"+modelname+".png"
lr = 0.0001

# %%

batch_size = 16
data_total_size = len(dataloader.train["indexes"])
steps = int(data_total_size/(batch_size))

epoch = 1

# %%

dict_data = []

################# training
for e in range(epoch) :

    tot_loss = 0
    t = tqdm(range(steps))
    for i in t :

        b_images, y = dataloader.get_batch(step=i, batch_size=batch_size)
        # print(b_images.shape)

        y_hat = model.forward(b_images)
        # print(y_hat)
        loss = model.calculate_loss(y_hat, y)
        model.backward(loss, y, lr=lr)
        tot_loss += loss
        loss_in_epoch = tot_loss/i
        t.set_description(str(loss_in_epoch))

        dict_data.append({
            "step" : i + steps*e,
            "loss" : loss_in_epoch
        })

        # break
    # dloader.epoch_end()


# %%

################# write logs
data_columns = ["step", "loss"]
try:
    with open(logfile, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
except IOError:
    print("I/O error")

# %%

################# testing
tot_test_step = int(len(dataloader.test["images"])/batch_size)
tot_acc = 0
confusion_matrix = np.zeros((10, 10))
best_img = [[] for i in range(10)]
for i in range(tot_test_step) :

    b_images, b_labels = dataloader.get_test(i, batch_size)

    output = model.forward(b_images)
    output = np.argmax(output, axis=1)
    b_labels = np.argmax(b_labels, axis=-1)
    
    acc = np.mean((output == b_labels)*1)
    output_answer = output.astype(np.int8)

    for b in range(output.shape[0]) :
        confusion_matrix[b_labels[b], output_answer[b]] += 1
        if b_labels[b] == output_answer[b] :
            best_img[output_answer[b]].append((output[b], b_images[b]))

    # confusion_matrix[]
    # print(output)
    tot_acc += acc

confusion_matrix_norm = confusion_matrix/np.sum(confusion_matrix, axis=0)

print("final accuracy is : ")
print(tot_acc/tot_test_step)


# %%

from matplotlib import pyplot as plt

fig = plt.figure()
plt.title("Prediction Result")
fig.tight_layout()
for v in best_img :
    ax = fig.add_subplot(1, 10, v[0][0]+1)
    ax.imshow(v[0][1])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.text(10, -5, "" + str(v[0][0]))
    fig.canvas.draw()

plt.savefig("images/bestimg_"+modelname+".png")
# plt.cla()


#%%

plt.plot([v["step"] for v in dict_data], [v["loss"] for v in dict_data])
plt.title("Loss graph for "+modelname)
plt.text(2000, 10, "total acc : " + str(tot_acc/tot_test_step)[:5])
plt.xlabel("Iteration")
plt.ylabel("Softmax Cross Entropy")

plt.savefig(lossfig_save)

# %%

df_cm = pd.DataFrame(confusion_matrix_norm, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.savefig("images/confusion_matrix_"+modelname+".png")

# %%

best_img[0][0][1].shape