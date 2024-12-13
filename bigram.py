import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 32 # this is represented by B
block_size = 8 # this is context size, represented by T
max_iters = 3000
learning_rate = 1e-2
eval_interval = 300
eval_iters = 200

# check for the device
def device_check():
    # Check for device support in priority order: CUDA > MPS > CPU
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Apple Metal Performance Shaders

    return device

device = device_check()
print(f"Using device: {device}")

# seed for reproducibility
torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()

# unique characters in the text, sorted
chars = sorted(list(set(text)))
vocab_size = len(chars) # this represented by C
# create a mapping of characters to integers and vice versa
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda str: [stoi[s] for s in str]
decode = lambda nums: [itos[n] for n in nums]
assert ''.join(decode(encode('hello'))) == 'hello'

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # encode
# now split it 90/10
n = int(0.9 * len(data))
# NOTE: not sampling throught the data for splitting. Why?
train_data, test_data = data[:n], data[n:]

# get batches of data from training sets
def get_batch(split):
    data = train_data if split == 'train' else test_data
    # generate a batch random indices one block shy of the end for x
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # the x values will now be pulled from data based on the indices
    # but pulled in a blcock size
    x = torch.stack([data[i:i+block_size]for i in ix])
    # y values will be shifted by one
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# interesting that this take a completely new set of training and val data
# and this makes the loss less noisy
@torch.no_grad() # telling torch that we won't call .backward()
def estimate_loss():
    out = {}
    model.eval() # set the model to the eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # reset the model to training mode
    return out

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # the forward pass
    def forward(self, x, y=None):
        # we are dealing in batches, x and y are of shape (B, T)
        logits = self.token_embedding_table(x) # (B, T, C)

        if y is None: # we are in inference/generation mode
            # we will simply return the logits
            loss = None
        else:
            # we are in training mode, we need to calculate the loss
            # we want to use cross-entropy loss
            # for this we need to reshape the logits to be 2D and y to be 1D
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)

        return logits, loss

    # generate text (as encoded tensors)
    def generate(self, init_x, num_tokens):
        idxs = init_x
        for _ in range(num_tokens):
            # get the logits for the current token
            logits, _ = self(idxs) # B, T, C
            logits = logits[:, -1, :] # B, C

            # keep taking the last logit and convert to probabilities
            # we do this since we've been appending to idxs
            probs = F.softmax(logits, dim=-1) # B, C
     
            # sample for the next token
            next_x = torch.multinomial(probs, num_samples=1) # B, 1

            # concat to the idxs
            idxs = torch.cat((idxs, next_x), dim=-1) # B, T+1
        return idxs

model = Bigram(vocab_size).to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

mean_losses = {'train': [], 'val': []}
for  iter in range(max_iters):
    # periodically evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Iter: {iter}, Train Loss: {losses['train']}, Validation Loss: {losses['val']}")
        mean_losses['train'].append(losses['train'].item())
        mean_losses['val'].append(losses['val'].item())

    # get a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)

    # normally we would call the loss function here, but forward already does

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_generated_text(model, num_tokens, seed_char=None):
    if seed_char is None:
        seed_char = torch.zeros((1, 1), dtype=torch.long)
    else:
        seed_char = torch.tensor([stoi[seed_char]], dtype=torch.long).unsqueeze(0)
    # be sure to move it to the device
    seed_char = seed_char.to(device)
    # generate text
    chars = model.generate(seed_char, num_tokens) # this will be a tensor
    chars = chars[0].tolist() # convert to a list
    return ''.join(decode(chars))

def plot_loss(mean_losses):
    plt.figure(figsize=(10, 5))
    plt.title('Mean Loss')
    plt.plot(mean_losses['train'], color='blue', label='train')
    plt.plot(mean_losses['val'], color='red', label='val')
    plt.legend()
    plt.show()

plot_loss(mean_losses)

# generate text
context = "\n"
output = get_generated_text(model, 100, seed_char=context)
print(output)
