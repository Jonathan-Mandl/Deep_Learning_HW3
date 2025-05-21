import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

# Configuration
SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
EMBED_DIM = 30
HIDDEN_DIM = 50

# Vocabulary: digits '1'-'9' and letters 'a','b','c','d'
VOCAB = [str(d) for d in range(1, 10)] + ['a', 'b', 'c', 'd']
char2idx = {c: i for i, c in enumerate(VOCAB)}
vocab_size = len(VOCAB)

def set_seeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                seq, label = line.strip().split('\t')
                label = int(label)
                idxs = [char2idx[c] for c in seq]
                self.data.append((torch.tensor(idxs, dtype=torch.long), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNNAcceptor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstmcell = nn.LSTMCell(embed_dim, hidden_dim)
        # Single-logit output for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # x: [batch, seq_len]
        batch_size, seq_len = x.size()
        embeds = self.embed(x)  # [batch, seq_len, embed_dim]
        hx = torch.zeros(batch_size, HIDDEN_DIM, device=embeds.device)
        cx = torch.zeros(batch_size, HIDDEN_DIM, device=embeds.device)
        for t in range(seq_len):
            hx, cx = self.lstmcell(embeds[:, t, :], (hx, cx))
        out = self.classifier(hx).squeeze(1)  # [batch]
        return out


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :lengths[i]] = s
    return padded, torch.tensor(labels, dtype=torch.long)


def main():
    set_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description="Train an RNN acceptor on sequence classification data."
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
        help="Path to training data file (tab-separated sequence and label)."
    )
    parser.add_argument(
        "--test_file", type=str, required=True,
        help="Path to test data file (tab-separated sequence and label)."
    )

    args=parser.parse_args()

    TRAIN_FILE = args.train_file
    TEST_FILE = args.test_file

    train_ds = SequenceDataset(TRAIN_FILE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_ds = SequenceDataset(TEST_FILE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = RNNAcceptor(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Use BCEWithLogitsLoss for single-logit binary setup
    criterion = nn.BCEWithLogitsLoss()

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        # --- Training phase ---
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # accumulate train accuracy
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct_train += (preds.cpu().view(-1) == labels.cpu().long()).sum().item()
            total_train += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train

        # --- Evaluation on test set ---
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for seqs, labels in test_loader:
                seqs, labels = seqs.to(device), labels.to(device).long()
                logits = model(seqs)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct_test += (preds.cpu().view(-1) == labels.cpu()).sum().item()
                total_test += labels.size(0)
        test_acc = 100.0 * correct_test / total_test

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
            f"Train time: {time.time() - start_time}"
        )

    train_time = time.time() - start_time
    print(f"Total training time: {train_time:.2f} sec")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
