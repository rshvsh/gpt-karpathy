import os
import time
import torch
import tiktoken

from model import GPT, GPTConfig
from args import parse_args, pretty_print, gen_cmd_line
from loaddata import DataLoaderRandom
from lr import get_lr
from evals import Evals

# -----------------------------------------------------------------------------
# Sample commandlines

# No evals, just train with ag_news dataset:
# python3 train_gpt2.py --dataset=data_ag_news --micro-batch-size=16 --max-steps=100 --warmup-steps=10 --val-loss-freq=-2 --hellaswag-freq=-2 --generate-freq=-2 --checkpoint-freq=-2 --graph-freq=-2

# torchrun with single GPU and all evals
# torchrun --standalone --nproc_per_node=1 train_gpt2.py --dataset=data_ag_news --micro-batch-size=16 --max-steps=100 --warmup-steps=10 --val-loss-freq=2 --hellaswag-freq=2 --generate-freq=2 --checkpoint-freq=2 --graph-freq=2

# torchrun with two GPUs and all evals except hellaswag (takes a lot of time)
# torchrun --standalone --nproc_per_node=2 train_gpt2.py --dataset=data_ag_news --micro-batch-size=16 --max-steps=100 --warmup-steps=10 --val-loss-freq=2 --hellaswag-freq=-2 --generate-freq=2 --checkpoint-freq=2 --graph-freq=2
# -----------------------------------------------------------------------------

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

args, parser = parse_args()
pretty_print(args)
cmdline = gen_cmd_line(args, parser)

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
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
    if master_process:
        print(f"--nproc_per_node={ddp_world_size}")
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
    print(f"{cmdline}")
    print(f"Total manually set batch size: {grad_accum_batch_size:_}")
    print(f"=> Calculated gradient accumulation steps: {grad_accum_steps:_}")

train_loader = DataLoaderRandom(process_rank=ddp_rank, ddp_world_size=ddp_world_size, split="train", dataset=args.dataset, args=args)
val_loader = DataLoaderRandom(process_rank=ddp_rank, ddp_world_size=ddp_world_size, split="val", dataset=args.dataset, args=args)

torch.set_float32_matmul_precision('high')

# create the log directory we will write checkpoints to and log to
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{args.log_file}")

# get some of the learning rate hyperparameters
weight_decay = args.weight_decay
max_lr = args.max_lr
warmup_steps = args.warmup_steps
max_steps = args.max_steps # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

# check if we are resuming from a checkpoint
checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
if len(checkpoint_files) > 0:
    # found checkpoints, load from the latest
    checkpoint_files = sorted(checkpoint_files)
    latest_checkpoint_file = checkpoint_files[-1]
    checkpoint_path = os.path.join(log_dir, latest_checkpoint_file)
    print(f"Restarting from checkpoint {checkpoint_path}") if master_process else None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # load model state
    model = GPT(checkpoint['config'])
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    # load optimizer state
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type, master_process=master_process)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # load step (which will also load learning rate)
    current_step = checkpoint['step'] + 1
    # load traning data state
    train_loader.set(checkpoint['train_loader'])
    print(f"Resuming training from step {current_step} with a validation loss of {checkpoint['val_loss']:.4f}") if master_process else None
else:
    print("No checkpoints found to resume training, starting from scratch") if master_process else None
    # create model
    gpt_config = GPTConfig(args) # instantiate from the arguments
    gpt_config.vocab_size = args.train_vocab_size # for training, override the gpt vocab size to be a good multiple of 2
    model = GPT(gpt_config)
    model.to(device)
    current_step = 0
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type, master_process=master_process)
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

# NOTE:~ from either of the above branches, we would have loaded the raw_model
# Now we need to compile it if we are not doing Hellaswag evaluations, AND
# We need to wrap it in DDP if we are doing DDP
use_compile = False # doesn't work with Hellaswag evaluation - TODO:~ why?
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# The above would have wrapped the model, so now we need to get the raw model  back again
def unwrap_model(model):
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module
    # Unwrap torch.compile
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model
raw_model = unwrap_model(model)

# setup the evaluations
evals = Evals(args=args,
              model=model,
              encoding=enc,
              process_rank=ddp_rank,
              num_processes=ddp_world_size,
              device_type=device_type,
              device=device,
              master_process=master_process,
              ddp=ddp,
              val_loader=val_loader,
              optimizer=optimizer,
              train_loader=train_loader)

for step in range(current_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    val_loss = 0.0

    # evaluate the val loss if configured
    if args.val_loss_freq > 0 and (step % args.val_loss_freq == 0 or last_step):
        val_loss = evals.calc_val_loss(step, last_step)

    # checkpoint the model if configured
    if step > 0 and args.checkpoint_freq > 0 and (step % args.checkpoint_freq == 0 or last_step):
        evals.checkpoint_model(step, last_step, val_loss)

    # evaluate hellaswag if configured
    if args.hellaswag_freq > 0 and (step % args.hellaswag_freq == 0 or last_step) and (not use_compile):
        evals.hellawag_eval(step)

    if args.graph_freq > 0 and (step > 0 and step % args.graph_freq == 0) or last_step:
        evals.print_simple_graph(step)
        # complex print only works when we evaluate hellaswag
        if args.hellaswag_freq > 0:
            evals.print_complex_graph(step)

    # generate output from the model if configured (except step 0, which is noise)
    if (args.generate_freq > 0) and ((step > 0 and step % args.generate_freq == 0) or last_step) and (not use_compile):
        evals.generate_text(step)

    # do one step of the optimization, go through all the microbatches in this grad eval step
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
