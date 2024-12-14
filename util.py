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

def create_output_dir():
    # if the output directory does not exist, create it
    if not os.path.exists(DEFAULT_DIR):
        os.makedirs(DEFAULT_DIR)

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

def write_outputs(mean_losses, more_text, ts):
    create_output_dir()
    write_text(more_text, f"output_text-{ts}.txt")
    write_loss(mean_losses, f"mean_loss-{ts}.json")
    plot_loss(f"mean_loss-{ts}.json", f"mean_loss-{ts}.png")
    return ts

def get_ts():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def save_model(model, fname):
    create_output_dir()
    fname = DEFAULT_DIR + "/" + fname
    torch.save(model.state_dict(), fname)
    return fname

def load_model(model, fname):
    # we get the fully qualified path from save_model above
    model.load_state_dict(torch.load(fname))
    return model
