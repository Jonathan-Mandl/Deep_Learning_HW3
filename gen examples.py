import random

# Configuration (no command-line arguments)
SEED = 42
SAMPLE_SIZE = 500       # number of positive/negative examples for pos_examples and neg_examples
TRAIN_SIZE = 20000      # number of positive and negative examples for training set
TEST_SIZE = 1000        # number of positive and negative examples for test set
DIGIT_LEN_RANGE = (1, 100)
LETTER_LEN_RANGE = (1, 100)
MAX_SEQ_LEN = 100       # maximum total length of each sequence


def generate_example(positive=True):
    """
    Generate a single sequence example whose total length <= MAX_SEQ_LEN.
    Positive pattern: [1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+
    Negative pattern: [1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+
    Each segment length is sampled then renormalized so the total stays within MAX_SEQ_LEN.
    """
    # Sample raw lengths for digit and letter segments
    raw_digit_lens = [random.randint(*DIGIT_LEN_RANGE) for _ in range(5)]
    raw_letter_lens = [random.randint(*LETTER_LEN_RANGE) for _ in range(4)]

    # Build segment type/order list
    if positive:
        order = ["digit", "a", "digit", "b", "digit", "c", "digit", "d", "digit"]
    else:
        order = ["digit", "a", "digit", "c", "digit", "b", "digit", "d", "digit"]

    # Interleave raw lengths according to order
    segs, raw_lengths = [], []
    di = li = 0
    for seg in order:
        if seg == "digit":
            raw_lengths.append(raw_digit_lens[di])
            segs.append(("digit", None))
            di += 1
        else:
            raw_lengths.append(raw_letter_lens[li])
            segs.append(("letter", seg))
            li += 1

    # Renormalize lengths to fit within MAX_SEQ_LEN
    total_raw = sum(raw_lengths)
    lengths = [max(1, int(l / total_raw * MAX_SEQ_LEN)) for l in raw_lengths]

    # Generate sequence parts
    parts = []
    for (seg_type, letter), length in zip(segs, lengths):
        if seg_type == "digit":
            parts.append(''.join(str(random.randint(1,9)) for _ in range(length)))
        else:
            parts.append(letter * length)

    return ''.join(parts)


def write_examples(filename, count, positive):
    with open(filename, 'w') as f:
        for _ in range(count):
            f.write(generate_example(positive) + '\n')


def write_labeled(filename, pos_count, neg_count):
    with open(filename, 'w') as f:
        for _ in range(pos_count):
            f.write(generate_example(True) + '\t1\n')
        for _ in range(neg_count):
            f.write(generate_example(False) + '\t0\n')


def main():
    random.seed(SEED)
    # Sample files
    write_examples('pos_examples', SAMPLE_SIZE, True)
    write_examples('neg_examples', SAMPLE_SIZE, False)
    # Train/Test sets
    write_labeled('train.txt', TRAIN_SIZE, TRAIN_SIZE)
    write_labeled('test.txt', TEST_SIZE, TEST_SIZE)


if __name__ == '__main__':
    main()
