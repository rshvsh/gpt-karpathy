import matplotlib.pyplot as plt
import os
import re
import json

DEFAULT_DIR = "output"

def write_loss(mean_losses, fname):
    # Dump the mean_losses to a JSON file
    fname = DEFAULT_DIR + "/" + fname
    with open(fname, 'w') as f:
        json.dump(mean_losses, f)
    
def plot_loss(loss_file, fname):
    loss_file = DEFAULT_DIR + "/" + loss_file
    fname = DEFAULT_DIR + "/" + fname
    # load the losses
    mean_losses = json.load(open(loss_file, 'r'))

    # plot the mean losses
    plt.figure(figsize=(10, 5))
    plt.title('Mean Loss')
    plt.plot(mean_losses['train'], color='blue', label='train')
    plt.plot(mean_losses['val'], color='red', label='val')
    plt.legend()

    # save the plot
    plt.savefig(fname)

def write_text(text, fname):
    fname = DEFAULT_DIR + "/" + fname
    with open(fname, 'w') as f:
        f.write(text)
