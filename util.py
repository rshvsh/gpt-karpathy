import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import torch

DEFAULT_DIR = "output"

# check for the device
def device_check():
    # Check for device support in priority order: CUDA > MPS > CPU
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Apple Metal Performance Shaders
    print(f"Using device: {device}")
    return device

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

def write_outputs(mean_losses, more_text):
    # if the output directory does not exist, create it
    if not os.path.exists(DEFAULT_DIR):
        os.makedirs(DEFAULT_DIR)

    # get the current timestamp as a string in YYYY-MM-DD-HH-MM-SS format
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("Writing files with timestamp:", ts)
    write_text(more_text, f"output_text-{ts}.txt")
    write_loss(mean_losses, f"mean_loss-{ts}.json")
    plot_loss(f"mean_loss-{ts}.json", f"mean_loss-{ts}.png")
