#%%

from models.ImageCaptioning import ImageCaptioning
from models.Cnn import Cnn
from dataloader.caption_data import Caption_data
from dataloader.mnist import MNIST_loader
import numpy as np
import csv
from pathlib import Path

#%%

model = Cnn()

dataloader = MNIST_loader()
dataloader.epoch_end()

modelname="ImageCaptioning"
logfile = "./log_"+modelname+".csv"
lossfig_save = "loss_graph_"+modelname+".png"
lr = 0.1

# %%

batch_size = 16
data_total_size = len(dataloader.train["indexes"])
steps = int(data_total_size/batch_size)

epoch = 1

# %%

dict_data = []

################# training
for e in range(epoch) :

    loss_in_epoch = 0
    for i in range(steps) :

        b_images, y = dataloader.get_batch(step=i, batch_size=batch_size)

        y_hat = model.forward(b_images, y)
        # print(y_hat)
        loss = model.calculate_loss(y_hat, y)
        print(loss)
        model.backward(loss, y, lr=lr)
        loss_in_epoch += loss/steps

        dict_data.append({
            "step" : i + steps*e,
            "loss" : loss
        })

        # break
    # dloader.epoch_end()



#%%
# i = 0
# for v in dict_data :
#     dict_data[i]["step"] = i
#     i += 1


# %%

################# write logs
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
tot_test_step = int(len(dloader.test["images"])/batch_size)
tot_acc = 0
confusion_matrix = np.zeros((10, 10))
best_img = [[] for i in range(10)]
for i in range(tot_test_step) :

    b_images, b_labels = dloader.get_test(i, batch_size)

    output_acc, output = model.evaluate(b_images, b_labels)
    
    output_answer = np.argmax(output, axis=1)

    for b in range(output.shape[0]) :
        confusion_matrix[b_labels[b], output_answer[b]] += 1
        if b_labels[b] == output_answer[b] :
            best_img[output_answer[b]].append((output[b], b_images[b]))


    # confusion_matrix[]
    # print(output)
    tot_acc += output_acc

print("final accuracy is : ")
print(tot_acc/tot_test_step)

#%%

from matplotlib import pyplot as plt


plt.plot([v["step"] for v in dict_data], [v["loss"] for v in dict_data])
plt.title("Loss graph by "+modelname)
plt.text(1200, 70, "total acc : " + str(tot_acc/tot_test_step)[:5])

plt.savefig(lossfig_save)
# plt.show()
