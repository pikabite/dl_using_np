#%%

from models.nn5model import NN5model
from models.nn5model2 import NN5model2
from models.cnnmodel import CNNmodel
from dataloader.mnist import MNIST_loader
import numpy as np
import csv


#%%

model = CNNmodel()
dloader = MNIST_loader()
modelname="cnnmodel"
logfile = "./log_"+modelname+".csv"
lossfig_save = "loss_graph_"+modelname+".png"
confmat_save = "confusion_matrix_"+modelname+".png"
top3images_save = "top3images_"+modelname+".png"
lr = 0.01

# %%

batch_size = 20
data_total_size = len(dloader.train["images"])
steps = int(data_total_size/batch_size)

data_columns = ['step','loss']

dict_data = []

epoch = 3

# %%

################# training
for e in range(epoch) :

    loss_in_epoch = 0
    for i in range(steps) :

        b_images, b_labels = dloader.get_batch(i, batch_size=batch_size)

        loss = model.train(b_images, b_labels, lr)

        print(loss)
        loss_in_epoch += loss/steps

        dict_data.append({
            "step" : i + steps*e,
            "loss" : loss
        })
    dloader.epoch_end()



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

#%%

# H = np.array([[1, 2, 3, 4],
#           [5, 6, 7, 8],
#           [9, 10, 11, 12],
#           [13, 14, 15, 16]])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.title("Confusion matrix for mnist dataset by "+modelname)
ax.grid(True)
plt.xlabel("prediction")
plt.ylabel("answer")
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
plt.imshow(confusion_matrix/10000)

plt.savefig(confmat_save)

# %%

for i in range(10) :
    class_imgs = best_img[i]

    
    sorted(class_imgs, key=lambda data: max(data[0]))

    # print(class_imgs[0])

#%%

# plt.imshow(best_img[0][0][1])

fig = plt.figure()
plt.title("Top 3 image by "+modelname)
plt.xlabel("classes")
plt.ylabel("top3 image")
plt.axis("off")
for i in range(1, 3*10+1) :
    ax = fig.add_subplot(10, 3, i)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.grid(False)
    plt.imshow(best_img[(i-1)//3][(i-1)%3][1])

# plt.show()
plt.savefig(top3images_save)



# %%
