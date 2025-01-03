# use this file to run various evals on the model
import torch
from torch.nn import functional as F
import torch.distributed as dist
from hellaswag import render_example, iterate_examples, get_most_likely_row
import os

class Evals:
    def __init__(self, args, model, encoding, process_rank, num_processes, device_type, device, master_process, ddp, val_loader):
        self.args = args
        self.model = model
        self.encoding = encoding
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device_type = device_type
        self.device = device
        self.master_process = master_process
        self.ddp = ddp
        self.val_loader = val_loader
        self.raw_model = self.model.module if self.ddp else self.model # always contains the "raw" unwrapped model
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

    def checkpoint_and_val_loss(self, step, last_step):
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

        if step > 0 and self.args.checkpoint_freq > 0 and (step % self.args.checkpoint_freq == 0 or last_step):
            checkpoint_path = os.path.join(self.log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': self.raw_model.state_dict(),
                'config': self.raw_model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            # TODO:~ you might also want to add optimizer.state_dict() and rng seeds etc., if you wanted to more exactly resume training
            torch.save(checkpoint, checkpoint_path)
