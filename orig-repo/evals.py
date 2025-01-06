# use this file to run various evals on the model
import torch
from torch.nn import functional as F
import torch.distributed as dist
from hellaswag import render_example, iterate_examples, get_most_likely_row
import os
import numpy as np
import matplotlib.pyplot as plt

class Evals:
    def __init__(self, args, model, encoding, process_rank, num_processes, device_type, device, master_process, ddp, val_loader, optimizer, train_loader):
        self.args = args
        self.model = model
        self.encoding = encoding
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device_type = device_type
        self.device = device
        self.master_process = master_process
        self.ddp = ddp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.raw_model = self.model.module if self.ddp else self.model # always contains the "raw" unwrapped model
        self.optimizer = optimizer
        self.log_dir = args.log_dir
        self.log_file = os.path.join(self.log_dir, f"{args.log_file}")

    def generate_text(self, step):
        self.model.eval()
        num_return_sequences = self.args.gen_text_num_samples
        max_length = self.args.gen_text_len
        tokens = self.encoding.encode(self.args.gen_text_prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(self.device)
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42 + self.process_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = self.encoding.decode(tokens)

            # Check for the '<|endoftext|>' token and truncate the text
            end_token = "<|endoftext|>"
            if end_token in decoded:
                decoded = decoded.split(end_token)[0]  # Only keep the text before the token
            print(f"step {step:5d} | rank {self.process_rank} sample {i}: {decoded}")

    def hellawag_eval(self, step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % self.num_processes != self.process_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # reduce the stats across all processes
        if self.ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=self.device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=self.device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if self.master_process:
            print(f"step {step:5d} | HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"{step:5d} hella {acc_norm:.4f}\n")

    def calc_val_loss(self, step, last_step):
        self.model.eval()
        self.val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            # TODO:~ be more accurate of the number of steps here by calculating how many validation batches are possible with the available tokens. For now this is ok. We don't want to omit data, or do more than one epoch
            val_loss_steps = self.args.val_loss_iters if len(self.val_loader.batches) * self.num_processes > self.args.val_loss_iters else len(self.val_loader.batches) * self.num_processes
            if self.master_process:
                print(f"step {step} Will run {val_loss_steps} validation loss calculations over {len(self.val_loader.batches)} validation batches") if self.args.debug_loader else None
            for i in range(val_loss_steps):
                print(f"Rank {self.process_rank}: step {step}: val_loss_step: {i}") if self.args.debug_loader else None
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                ##### removed loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            val_loss_accum /= val_loss_steps # diff from Andrej - less prone to floating point errors: https://github.com/karpathy/build-nanogpt/pull/19/files
        if self.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.master_process:
            print(f"step {step:5d} | validation loss: {val_loss_accum.item():.4f} calculated in {val_loss_steps} val_loss_steps")
            with open(self.log_file, "a") as f:
                f.write(f"{step:5d} val {val_loss_accum.item():.4f}\n")
        return val_loss_accum.item()

    def checkpoint_model(self, step, last_step, val_loss_accum):
        if val_loss_accum == 0.0:
            # we need the validation loss so that we know where we are in the process
            val_loss_accum = self.calc_val_loss(step, last_step)
            if self.master_process:
                print(f"Calculated validation loss = {val_loss_accum:.6f} for checkpointing")

        if self.master_process:
            # use a temp path to ensure atomicity of the save operation to the final file name
            tmp_checkpoint_path = os.path.join(self.log_dir, f"tmp.ptt")

            # we need to know where we are in the dataloader
            train_loader_checkpoint = {
                'current_shard_index': self.train_loader.current_shard_index,
                'current_batch_index': self.train_loader.current_batch_index,
                'batches': self.train_loader.batches,
                # 'tokens': self.train_loader.tokens, # we don't need to save the tokens, we can load them
                # TODO:~ check for any off-by-one errors in the above
            }
            checkpoint = {
                'model': self.raw_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.raw_model.config,
                'step': step,
                'val_loss': val_loss_accum,
                'train_loader': train_loader_checkpoint,        
            }
            torch.save(checkpoint, tmp_checkpoint_path)
            checkpoint_path = os.path.join(self.log_dir, f"model_{step:05d}.pt")
            os.rename(tmp_checkpoint_path, checkpoint_path)

    def print_simple_graph(self, iter):
        sz = "124M"

        # # load the log file
        with open(self.log_file, "r") as f:
            lines = f.readlines()

        # parse the individual lines, group by stream (train,val,hella)
        streams = {}
        for line in lines:
            step, stream, val = line.strip().split()
            if stream not in streams:
                streams[stream] = {}
            streams[stream][int(step)] = float(val)

        # convert each stream from {step: val} to (steps[], vals[])
        # so it's easier for plotting
        streams_xy = {}
        for k, v in streams.items():
            # get all (step, val) items, sort them
            xy = sorted(list(v.items()))
            # unpack the list of tuples to tuple of lists
            streams_xy[k] = list(zip(*xy))

        xs, ys = streams_xy["train"] # training loss
        ys = np.array(ys)
        plt.plot(xs, ys, label=f'({sz}) train loss')
        xs, ys = streams_xy["val"] # validation loss
        plt.plot(xs, ys, label=f'({sz}) val loss')
        plt.legend()
        plt.title(f"Loss Graph for {sz}")
        plt.xlabel("Steps")
        plt.ylabel("Loss")

        # Save the figure
        output_file = os.path.join(self.log_dir, f"{self.args.dataset}_simple_loss_graph_{iter:04d}.png")
        plt.savefig(output_file)
        plt.close()

    def print_complex_graph(self, iter):
        sz = "124M"
        hella2_baseline = 0.294463
        hella3_baseline = 0.337
        loss_baseline = 3.2924
    
        # load the log file
        with open(self.log_file, "r") as f:
            lines = f.readlines()

        # parse the individual lines, group by stream (train,val,hella)
        streams = {}
        for line in lines:
            step, stream, val = line.strip().split()
            if stream not in streams:
                streams[stream] = {}
            streams[stream][int(step)] = float(val)

        # convert each stream from {step: val} to (steps[], vals[])
        # so it's easier for plotting
        streams_xy = {}
        for k, v in streams.items():
            # get all (step, val) items, sort them
            xy = sorted(list(v.items()))
            # unpack the list of tuples to tuple of lists
            streams_xy[k] = list(zip(*xy))

        # create figure
        plt.figure(figsize=(16, 6))

        # Panel 1: losses: both train and val
        plt.subplot(121)
        xs, ys = streams_xy["train"] # training loss
        ys = np.array(ys)
        plt.plot(xs, ys, label=f'nanogpt ({sz}) train loss')
        print("Min Train Loss:", min(ys))
        xs, ys = streams_xy["val"] # validation loss
        plt.plot(xs, ys, label=f'nanogpt ({sz}) val loss')
        # horizontal line at GPT-2 baseline
        if loss_baseline is not None:
            plt.axhline(y=loss_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint val loss")
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.yscale('log')
        plt.ylim(top=15.0)
        plt.legend()
        plt.title("Loss")
        print("Min Validation Loss:", min(ys))

        # Panel 2: HellaSwag eval
        plt.subplot(122)
        xs, ys = streams_xy["hella"] # HellaSwag eval
        ys = np.array(ys)
        plt.plot(xs, ys, label=f"nanogpt ({sz})")
        # horizontal line at GPT-2 baseline
        if hella2_baseline:
            plt.axhline(y=hella2_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint")
        if hella3_baseline:
            plt.axhline(y=hella3_baseline, color='g', linestyle='--', label=f"OpenAI GPT-3 ({sz}) checkpoint")
        plt.xlabel("steps")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("HellaSwag eval")
        print("Max Hellaswag eval:", max(ys))

        # Save the figure
        output_file = os.path.join(self.log_dir, f"{self.args.dataset}_complex_loss_graph_{iter:04d}.png")
        plt.savefig(output_file)
        plt.close()
