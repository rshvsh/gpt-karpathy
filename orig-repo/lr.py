import math

# use this to get the learning rate
def get_lr(it, max_lr, max_steps, weight_decay, warmup_steps):
    min_lr = max_lr * weight_decay
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

if __name__ == "__main__":
    # use this to check the learning rate decay as a function of iteration
    from args import parse_args
    import matplotlib.pyplot as plt

    args = parse_args()
    lr = [[i, get_lr(i, args.max_lr, args.max_steps, args.weight_decay, args.warmup_steps)] for i in range(args.max_steps)]

    # Extract i (x-axis) and lr (y-axis)
    x = [point[0] for point in lr]
    y = [point[1] for point in lr]

    # Plot
    plt.plot(x, y, marker="o", linestyle="-", color="b")  # Line plot with markers
    plt.title("Learning Rate vs Iteration")
    plt.xlabel("Iteration (i)")
    plt.ylabel("Learning Rate (lr)")
    plt.grid(True)
    plt.savefig("tmp.png", dpi=300)
    plt.show()