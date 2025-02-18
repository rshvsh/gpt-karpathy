import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    
    parser.add_argument("--dataset", type=str, default="edu_fineweb10B", help="Training dataset name")
    parser.add_argument("--grad-accum-batch-size", type=int, default=524288, help="Batch size for gradient accumulation")
    parser.add_argument("--micro-batch-size", type=int, default=64, help="Micro batch size")
    parser.add_argument("--sequence-length", type=int, default=1024, help="Sequence length")

    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max-lr", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=715, help="Warmup step for the first epoch")
    parser.add_argument("--max-steps", type=int, default=19073, help="Maximum steps in one epoch")

    parser.add_argument("--gpt-block-size", type=int, default=1024, help="Default block size in the GPT model. Should be equal to the sequence length")
    parser.add_argument("--gpt-vocab-size", type=int, default=50257, help="Vocab size default in the GPT model. Sometimes rounded up to be a nice even number during training")
    parser.add_argument("--train-vocab-size", type=int, default=50304, help="Vocab size used during training. Should be a good multiple of 2, rounded up above the GPT vocab size")

    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num-embds", type=int, default=768, help="Number of embeddin dimensions")

    parser.add_argument("--val-loss-freq", type=int, default=25, help="How often to calculate validation loss")
    parser.add_argument("--val-loss-iters", type=int, default=20, help="How many iterations to calculate validation loss")
    parser.add_argument("--hellaswag-freq", type=int, default=250, help="How often to evaluate hellaswag")
    parser.add_argument("--generate-freq", type=int, default=250, help="How often to generate text")
    parser.add_argument("--checkpoint-freq", type=int, default=5000, help="How often to checkpoint the model")
    parser.add_argument("--graph-freq", type=int, default=500, help="How often to create a graph of the val loss")

    parser.add_argument("--debug-loader", action='store_true', help="Prints messages with different process rank indices to diagnose loader issues.")
    parser.add_argument("--log-dir", type=str, default="log", help="The log directory")
    parser.add_argument("--log-file", type=str, default="log.txt", help="The log file name")

    parser.add_argument("--gen-text-prompt", type=str, default="Hello, I'm a language model,", help="The default prompt for text generation")
    parser.add_argument("--gen-text-num-samples", type=int, default=4, help="The number of samples to generate")
    parser.add_argument("--gen-text-len", type=int, default=32, help="The maximum length of the text to be generated")

    args = parser.parse_args()
    return args, parser

def pretty_print(args):
    print("\n")
    print(f"Training dataset:{args.dataset}")

    print(f"\nThese affect the memory the training data batches need:")
    print(f"\tGrad accumulation batch     {args.grad_accum_batch_size:_}")
    print(f"\tMicro batch size            {args.micro_batch_size:_}")
    print(f"\tSequence length             {args.sequence_length:_}")

    print(f"\nThese affect model size:")
    print(f"\tBlock size                  {args.gpt_block_size:_}")
    print(f"\tGPT vocab size              {args.gpt_vocab_size:_}")
    print(f"\tTrain vocab size            {args.train_vocab_size:_}\t(should be even and as close to a power of 2 as possible)")
    print(f"\tNum layers                  {args.num_layers}")
    print(f"\tNum heads                   {args.num_heads}")
    print(f"\tNum embedding dimensions    {args.num_embds}")

    print(f"\nThese affect the learning rate decay:")
    print(f"\tWeight decay                {args.weight_decay:.6f}")
    print(f"\tMaximum learning rate       {args.max_lr:.8f}")
    print(f"\tWarmup steps                {args.warmup_steps:_}")
    print(f"\tMax steps                   {args.max_steps:_}")

    print(f"\nThese are used to change the log file location:")
    print(f"\tThe log directory           {args.log_dir}")
    print(f"\tThe log file                {args.log_file}")

    print(f"\nThese affect frequency of various validations/checkpoints:")
    print(f"\tVal loss frequency          {args.val_loss_freq:_}\t\t(<0 to disable)")
    print(f"\tVal loss iterations         {args.val_loss_iters:_}")
    print(f"\tHellaswag frequency         {args.hellaswag_freq:_}\t\t(<0 to disable)")
    print(f"\tText generation frequency   {args.generate_freq:_}\t\t(<0 to disable)")
    print(f"\tModel checkpoint frequency  {args.checkpoint_freq:_}")
    print(f"\tModel graph frequency       {args.graph_freq:_}")

    print(f"\nThese affect text generation from a provided prompt:")
    print(f"\tThe text prompt             {args.gen_text_prompt}")
    print(f"\tNum samples generated       {args.gen_text_num_samples}")
    print(f"\tMax len of samples          {args.gen_text_len}")

    print(f"\nThis is used to debug dataloader issues with multiple GPUs:")
    print(f"\tDebug loader messages       {args.debug_loader}")
    print("\n")

def gen_cmd_line(args, parser):
    command = []
    # Get the default values
    defaults = {action.dest: action.default for action in parser._actions}
    
    # Compare args with defaults
    for key, value in vars(args).items():
        if value != defaults[key]:
            if isinstance(value, bool):
                if value:  # Add flag only if True
                    command.append(f"--{key.replace('_', '-')}")
            else:
                command.append(f"--{key.replace('_', '-')}={value}")
    
    return " ".join(command)

if __name__ == "__main__":
    args, parser = parse_args()
    pretty_print(args)
    cmd = gen_cmd_line(args, parser)
    print(f"Non-default command-line arguments:\n{cmd}")

    import sys
    sys.exit(0)
    print("""
          NOTE:
          grad_accum_batch_size MUST be a multiple of B * T * ddp_world_size

          - Seems like a large grad_accum_batch_size is good, because it prevents the model from updating too frequently
          - Seems like the norms don't fluctuate too much with a well-sized grad_accum_batch_size
          - grad_accum_batch_size = 524288 is a good value (used in the video) 
          - It should be a good even multiple of 2

          max_steps should be calculated based on how many you expect in the dataset

          EXAMPLE1:
          - 6.3M tokens (shard size = 100k tokens)
          - grad_accume_batch_size = 524288
          => max_steps = 6.3M / 524288 = 12 steps [1 epoch]

          - B = 16
          - T = 1024
          - world_size = 1
          => num_micro_steps = 524288 / (16 * 1024 * 1) = 32 steps

          useful to know num microsteps per shard
          - 100k tokens / (16 * 1024 * 1) = 6 steps

          EXAMPLE2: (karpathy's video settings)
          - 10B tokens (shard size = 100M tokens)
          - grad_accume_batch_size = 524288
          => max_steps =10B / 524288 = 19073 steps [1 epoch]

          - B = 64
          - T = 1024
          - world_size = 8
          => num_micro_steps = 524288 / (64 * 1024 * 8) = 1 steps

          useful to know num microsteps per shard
          - 100M tokens / (64 * 1024 * 8) = 19 steps          

          """)
