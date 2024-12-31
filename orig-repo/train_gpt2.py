import os
import time
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import tiktoken

from model import GPT, GPTConfig
from args import parse_args
from loaddata import DataLoaderLite, DataLoaderRandom
from lr import get_lr
from evals import generate_text, calc_val_loss, hellaswag_eval

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
#
# here is a sample commandline with args that exercises a lot of options
# torchrun --standalone --nproc_per_node=1 train_gpt2.py --micro-batch-size=16 --val-loss-freq=2 --hellaswag-freq=2 --dataset=data_ag_news --max-steps=100 --warmup-steps=10 --generate-freq=2 --checkpoint-freq=2
# python3 train_gpt2.py --dataset=data_ag_news --micro-batch-size=16 --val-loss-freq=2 --hellaswag-freq=-2 --max-steps=100 --warmup-steps=10 --generate-freq=2 --checkpoint-freq=2

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

args = parse_args()
max_arg_length = max(len(arg) for arg in vars(args).keys())
for arg, value in vars(args).items():
    print(f"{arg:<{max_arg_length}}: {value}")

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

grad_accum_batch_size = args.grad_accum_batch_size # 2**19, 524288 ~0.5M, in number of tokens
B = args.micro_batch_size # micro batch size # original was 64... needed to reduce due to CUDA out of memory
T = args.sequence_length # 1024 sequence length
assert grad_accum_batch_size % (B * T * ddp_world_size) == 0, "make sure grad_accum_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = grad_accum_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total manually set batch size: {grad_accum_batch_size}")
    print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderRandom(B=B, T=T, process_rank=ddp_rank, ddp_world_size=ddp_world_size, split="train", dataset=args.dataset, master_process=master_process)
val_loader = DataLoaderRandom(B=B, T=T, process_rank=ddp_rank, ddp_world_size=ddp_world_size, split="val", dataset=args.dataset, master_process=master_process)

torch.set_float32_matmul_precision('high')

# create model
gpt_config = GPTConfig(args) # instantiate from the arguments
gpt_config.vocab_size = args.train_vocab_size # for training, override the gpt vocab size to be a good multiple of 2
model = GPT(gpt_config)
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

weight_decay = args.weight_decay
max_lr = args.max_lr
warmup_steps = args.warmup_steps
max_steps = args.max_steps # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type, master_process=master_process)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss and checkpoint the model
    if args.val_loss_freq > 0 and (step % args.val_loss_freq == 0 or last_step):
        calc_val_loss(model, val_loader, device, device_type, ddp, master_process, log_dir, log_file, step, args.checkpoint_freq, raw_model, last_step, args.val_loss_iters)

    # once in a while evaluate hellaswag
    if args.hellaswag_freq > 0 and (step % args.hellaswag_freq == 0 or last_step) and (not use_compile):
        hellaswag_eval(model, device, device_type, ddp, master_process, log_file, ddp_world_size, ddp_rank, step)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % args.generate_freq == 0) or last_step) and (not use_compile):
        generate_text(model, enc, device, ddp_rank, device_type)

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # print(f"micro_step: {micro_step}")
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step, max_lr, max_steps, weight_decay, warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
