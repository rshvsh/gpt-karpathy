import numpy as np
import os
import torch

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# We want to pick the shards randomly without replacement
# Then within the shard, we want to pick the micro-batches randomly without replacement
import random
class DataLoaderRandom:

    def __init__(self, process_rank, ddp_world_size, split, dataset, args):
        print(f"Rank {process_rank}: DataLoaderRandom for {split}")
        self.B = args.micro_batch_size
        self.T = args.sequence_length
        self.process_rank = process_rank
        self.num_processes = ddp_world_size
        self.split = split
        self.args = args
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = dataset
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        random.seed(1957) # the seed is important so that all processes pick the same shards
        random.shuffle(shards)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"Rank {process_rank}: Split {split}: Found {len(shards)} shards")

        # this is the size of the batch being pulled
        self.dataloader_batch_size = self.B * self.T * ddp_world_size
        print(f"Rank {self.process_rank}: Split {split}: Total data loader batch size: {self.dataloader_batch_size:_}")
        
        self.reset()

    # reset with current_shard_index = 0 means starting a new epoch
    def reset(self, current_shard_index=0):
        self.current_shard_index = current_shard_index # start at shard index zero
        self.current_batch_index = 0 # keeps track of the batch index within the current shard
        self.batches = [] # list of batch indices to pull from the current shard

        # load tokens for the shard
        self.tokens = load_tokens(self.shards[self.current_shard_index])
        num_tokens = len(self.tokens)
        num_batches = (num_tokens - 1) // self.dataloader_batch_size # added the -1 so that we have an extra token for the last batch

        print(f"Rank {self.process_rank}: Split {self.split}: reset(): shard index {self.current_shard_index}:{self.shards[self.current_shard_index]} has {num_tokens:_} tokens and {num_batches:_} batches") if self.args.debug_loader else None
        if num_batches == 0:
            # this shard is too small, skip it and go to the next one
            print(f"Rank {self.process_rank}: Split {self.split}: reset(): Shard is too small, skipping shard index {self.current_shard_index}:{self.shards[self.current_shard_index]}") if self.args.debug_loader else None
            next_shard_index = 0 if self.current_shard_index + 1 >= len(self.shards) else self.current_shard_index + 1
            # TODO:~ at some point check for infinite loop here, but for now we assume there is always a valid shard
            self.reset(current_shard_index=next_shard_index)
            return

        # The following approach shuffles the batches in the shard. 
        # An alternate way is to shuffle the documents (separated by |<endoftext>|) in the shard
        # Then we wouldn't have to worry about the current_batch_index
        # NOTE:~ regardless, I think we should still shuffles the shards themselves
        self.batches = list(range(num_batches))
        random.seed(1957) # the seed is important so that all processes pick from the same batches
        random.shuffle(self.batches)

    def next_batch(self):
        B, T = self.B, self.T
        start_position = B * T * self.num_processes * self.current_batch_index + B * T * self.process_rank
        print(f"Rank: {self.process_rank}: Split {self.split}: next_batch(): start_position={start_position}") if self.args.debug_loader else None
        buf = self.tokens[start_position : start_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        self.current_batch_index += 1 # advance the batch index for this process
        if self.current_batch_index >= len(self.batches):
            # we need to advance to the next shard
            self.current_shard_index += 1
            if self.current_shard_index >= len(self.shards):
                # we are out of shards, reset to the beginning
                if self.split == 'train' or self.args.debug_loader:
                    print(f"Rank {self.process_rank}: Split {self.split}: Shard index {self.current_shard_index}: Resetting to the first shard index, we've gone through one epoch of data")
                self.reset()
            else:
                # load tokens for the next shard
                self.reset(self.current_shard_index)
                self.tokens = load_tokens(self.shards[self.current_shard_index])

        return x, y

if __name__ == "__main__":
    from args import parse_args, pretty_print
    args = parse_args()
    pretty_print(args)
    print("micro batch size:", args.micro_batch_size)
    print("sequence length:", args.sequence_length)
    world_size = 1
    process_num = 0
    num_shards = 63
    shard_len = 100_000 # tokens
    num_batches_per_shard = shard_len // (args.micro_batch_size * args.sequence_length * world_size)
    iters = num_shards * num_batches_per_shard + 10

    dl = DataLoaderRandom(process_num, world_size, "val", args.dataset, args)
    print(f"Running {iters} iterations")
    print("iter, shard index, batch index")
    for i in range(iters):
        print(f"{i}, {dl.current_shard_index}, {dl.current_batch_index}")
        x, y = dl.next_batch()

    # This checks for skipping small shards within 10 iterations with world size 8
    # python loaddata.py --dataset=data_ag_news --micro-batch-size=45 --sequence-length=256

    # This checks for circling one epoch with 64 iterations with world size 8
    # python loaddata.py --dataset=data_ag_news --micro-batch-size=32 --sequence-length=256

    # This is the actual training params with world size 1
    # python loaddata.py --dataset=data_ag_news --micro-batch-size=16
    # python3 loaddata.py --micro-batch-size=16 --val-loss-freq=2 --hellaswag-freq=-2 --dataset=data_ag_news --max-steps=100 --warmup-steps=10 --generate-freq=2 --checkpoint-freq=2
