import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    
    parser.add_argument("--dataset", type=str, default="edu_fineweb10B", help="Training dataset name")
    parser.add_argument("--total-batch-size", type=int, default=524288, help="Total batch size")
    parser.add_argument("--micro-batch-size", type=int, default=64, help="Micro batch size")
    parser.add_argument("--sequence-length", type=int, default=1024, help="Sequence length")

    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max-lr", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=715, help="Warmup step for the first epoch")
    parser.add_argument("--max-steps", type=int, default=19073, help="Maximum steps in one epoch")

    parser.add_argument("--gpt-block-size", type=int, default=1024, help="Default block size in the GPT model. Should be equal to the sequence length")
    parser.add_argument("--gpt-vocab-size", type=int, default=50257, help="Vocab size default in the GPT model. Sometimes rounded up to be a nice even number during training")
    parser.add_argument("--train-vocab-size", type=int, default=50304, help="Vocab size default in the GPT model. Sometimes rounded up to be a nice even number during training")

    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num-embds", type=int, default=768, help="Number of embeddin dimensions")

    parser.add_argument("--val-loss-freq", type=int, default=25, help="How often to calculate validation loss")
    parser.add_argument("--hellaswag-freq", type=int, default=250, help="How often to evaluate hellaswag")
    parser.add_argument("--generate-freq", type=int, default=250, help="How often to generate text")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"Training for {args.dataset} dataset")
    print(f"Total batch {args.total_batch_size}")
    print(f"Micro batch size {args.micro_batch_size}")
    print(f"Sequence length {args.sequence_length}")

    print(f"Weight decay {args.weight_decay:.6f}")
    print(f"Maximum learning rate {args.max_lr:.8f}")
    print(f"Warmup steps {args.warmup_steps}")
    print(f"Max steps {args.max_steps}")

    print(f"Block size {args.gpt_block_size}")
    print(f"GPT vocab size {args.gpt_vocab_size}")
    print(f"Train vocab size {args.train_vocab_size}")

    print(f"Num layers {args.num_layers}")
    print(f"Num heads {args.num_heads}")
    print(f"Num embedding dimensions {args.num_embds}")

    print(f"Val loss frequency {args.val_loss_freq}")
    print(f"Hellaswag frequency {args.hellaswag_freq}")
    print(f"Text generation frequency {args.generate_freq}")
