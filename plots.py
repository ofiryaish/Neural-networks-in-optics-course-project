import os, time, scipy.io
import numpy as np
import rawpy
import glob
import re
import matplotlib.pyplot as plt
home_dir = '/data/yaishof/NN_project/DeepImageDenoising/'
def plot_epochs_loss (NUM_OF_ITERATIONS, file_name, title, label=''):
    with open(home_dir+file_name) as f:
        lines = f.readlines()
        epoch_loss_list = []
        current_epoch_loss = 0
        interation_num = 0 
        for line in lines:
            result = re.search('Loss=(.*) Time=', line)
            current_loss = float(result.group(1))
            current_epoch_loss = current_epoch_loss + current_loss
            if (interation_num == NUM_OF_ITERATIONS-1):
                epoch_loss_list.append(current_epoch_loss/NUM_OF_ITERATIONS) #average
                current_epoch_loss = 0
                interation_num = 0
            else:
                interation_num = interation_num + 1

        plt.plot(range(len(epoch_loss_list)), epoch_loss_list, label=label)
        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel("Epoch")
        #plt.savefig(title.replace('\n','')+".png")
        #plt.cla()



# NUM_OF_ITERATIONS = 162
# file_name = "log_sony_regular.txt"
# title = "Loss as function of epoch - Sony dataset"
# plot_epochs_loss (NUM_OF_ITERATIONS, file_name, title)

NUM_OF_ITERATIONS = 136
file_name = "log_fuji_upNeighbor.txt"
title = "Loss as function of epoch - Fuji dataset\n Upsampling using nearest neighbor"
plot_epochs_loss (NUM_OF_ITERATIONS, file_name, title, 'nearest neighbor')

NUM_OF_ITERATIONS = 136
file_name = "log_fuji_upConv.txt"
title = "Loss as function of epoch - Fuji dataset\n Upsampling using transposed convolution"
plot_epochs_loss (NUM_OF_ITERATIONS, file_name, title, "transposed convolution")
plt.legend(loc="upper right")
plt.savefig("test.png")