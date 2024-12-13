import torch
import torch.nn as nn
import torch.nn.functional as F
import util as u
import sys

# hyperparameters for running on my macbook pro, set big = False
big = True if (len(sys.argv) > 1 and sys.argv[1] == 'big') else False
batch_size = 32 if not big else 64        # this is represented by B
block_size = 8  if not big else 256       # this is context size, represented by T
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3 if not big else 3e-4
device = u.device_check()
eval_iters = 200
n_embed = 32 if not big else 384          # this is represented by C
n_head = 4 if not big else 6
n_layer = 3 if not big else 6
dropout = 0.2
# ------------

torch.manual_seed(1337) # for reproducibility

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()

# unique characters in the text, sorted
chars = sorted(list(set(text)))
vocab_size = len(chars) # this represented by C
# create a mapping of characters to integers and vice versa
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda str: [stoi[s] for s in str]    # string to integer
decode = lambda nums: ''.join([itos[n] for n in nums]) # integer to string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # encode
# now split it 90/10
n = int(0.9 * len(data))
# NOTE: not sampling throught the data for splitting. Why?
train_data, val_data = data[:n], data[n:]

# get batches of data from training sets
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # generate a batch random indices one block shy of the end for x
    ix = torch.randint(len(data) - block_size, (batch_size,))      # get some indices
    x = torch.stack([data[i:i+block_size]for i in ix])             # pull some values
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # shift y by 1
    x, y = x.to(device), y.to(device)                              # move to device
    return x, y

# estimate losses on completely different train and val samples
@torch.no_grad() # telling torch that we won't call .backward()
def estimate_loss():
    out = {}
    model.eval()             # set the model to the eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()            # reset the model to training mode
    return out

# a single head of attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x) # B, T, hs
        q = self.query(x) # B, T, hs
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** (-0.5) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)                # (B, T, T)
        wei = self.dropout(wei)
        # now perform weighted aggregation of values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
# multi-head attention
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
# feed forward
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
# blocks
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)        
        self.blocks = nn.Sequential( *[Block(n_embed, n_head) for _ in range(n_layer)] )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # the forward pass
    def forward(self, x, y=None):
        # we are dealing in batches, x and y are of shape (B, T)
        B, T = x.shape

        tok_embed = self.token_embedding_table(x) # (B, T, C)
        pos_embed = self.pos_embed(torch.arange(T, device=device)) # (T, C) ... T broadcasted to C
        # add positional info to the tokens
        x = tok_embed + pos_embed # (B, T, C) ... T, C broadcasted to B
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
 
        if y is None: # we are in inference/generation mode
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
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idxs to block size becaue of positional encoding
            idxs_cropped = idx[:, -block_size:]
            # get the logits for the current token
            logits, _ = self(idxs_cropped)   # (B, T, C)
            logits = logits[:, -1, :]        # (B, C)
            # keep taking the last logit and convert to probabilities
            # we do this since we've been appending to idxs
            probs = F.softmax(logits, dim=-1) # B, C
            # sample for the next token
            next_x = torch.multinomial(probs, num_samples=1) # B, 1
            # concat to the idxs
            idx = torch.cat((idx, next_x), dim=1) # B, T+1
        return idx

model = GPT().to(device)
# print the number of parameters
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

mean_losses = {'train': [], 'val': []}
for  iter in range(max_iters):
    # periodically evaluate the loss on train and val sets
    if iter == 0 or ((iter + 1) % eval_interval) == 0:
        losses = estimate_loss()
        print(f"Iter: {iter + 1}, Train Loss: {losses['train']:.4f}, Validation Loss: {losses['val']:.4f}")
        mean_losses['train'].append(losses['train'].item())
        mean_losses['val'].append(losses['val'].item())

    # get a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad()     # zero the gradients
    loss.backward()           # backprop gradients
    optimizer.step()          # update the weights

# ------------
# save the model at the current timestampa
ts = u.get_ts() # timestap for outputs
print("Writing files with timestamp:", ts)

model_file = u.save_model(model, f"gpt_model-{ts}.gpt") # save model
model = u.load_model(GPT().to(device), model_file)     # see if model loads

# generate from the loaded model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
more_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())

# write the outputs
u.write_outputs(mean_losses, more_text, ts)
print("Generated text:")
print(more_text)
# ------------