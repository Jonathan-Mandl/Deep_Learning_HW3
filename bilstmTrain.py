import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class SequenceDataset(Dataset):
    def __init__(self, data_file, word_to_ix, tag_to_ix, repr_type='a'):
        self.sentences = []
        self.labels = []
        self.word_strs = []
        with open(data_file) as f:
            sentence = []
            tags = []
            sentence_str = []
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        self.sentences.append(sentence)
                        self.labels.append(tags)
                        self.word_strs.append(sentence_str)
                        sentence = []
                        tags = []
                        sentence_str = []
                else:
                    word, tag = line.split()
                    sentence.append(word_to_ix.get(word, word_to_ix['<UNK>']))
                    tags.append(tag_to_ix[tag])
                    sentence_str.append(word)
            if sentence:
                self.sentences.append(sentence)
                self.labels.append(tags)
                self.word_strs.append(sentence_str)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx]), self.word_strs[idx]

def pad_collate(batch):
    sentences, labels, word_strs = zip(*batch)
    lengths = [len(s) for s in sentences]
    padded_sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return padded_sentences, padded_labels, lengths, word_strs

class CharBiLSTMEmbedder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, char_hidden_dim):
        super(CharBiLSTMEmbedder, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        self.char_lstm = nn.LSTM(char_emb_dim, char_hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, word_list, char_to_ix):
        reps = []
        for word in word_list:
            char_ids = [char_to_ix.get(c, char_to_ix['<UNK>']) for c in word]
            char_tensor = torch.tensor(char_ids).unsqueeze(0)  # shape: (1, len)
            char_embs = self.char_embedding(char_tensor)
            _, (hn, _) = self.char_lstm(char_embs)
            rep = torch.cat((hn[0], hn[1]), dim=1)  # shape: (1, 2*hidden)
            reps.append(rep.squeeze(0))
        return torch.stack(reps)

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, char_vocab_size=None, repr_type='a', emb_dim=30, hidden_dim=50,
                 dropout=0.3, prefix_size=None, suffix_size=None):
        super(BiLSTMTagger, self).__init__()
        self.repr_type = repr_type
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)

        if repr_type == 'b':
            self.char_embedder = CharBiLSTMEmbedder(char_vocab_size, 15, emb_dim // 2)
        if repr_type == 'c':
            self.prefix_embedding = nn.Embedding(prefix_size, emb_dim)
            self.suffix_embedding = nn.Embedding(suffix_size, emb_dim)

        if repr_type == 'd':
            self.char_embedder = CharBiLSTMEmbedder(char_vocab_size, 15, emb_dim // 2)
            self.combine_proj = nn.Linear(emb_dim + emb_dim, emb_dim)

        self.lstm1 = nn.LSTM(emb_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(2 * hidden_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, x, lengths, word_strs=None, char_to_ix=None, prefix_to_ix=None, suffix_to_ix=None):
        if self.repr_type == 'a':
            emb = self.embedding(x)
        elif self.repr_type == 'b':
            batch_reprs = []
            for sent in word_strs:
                word_reprs = self.char_embedder(sent, char_to_ix)
                batch_reprs.append(word_reprs)
            emb = nn.utils.rnn.pad_sequence(batch_reprs, batch_first=True)
        elif self.repr_type == 'd':
            emb_word = self.embedding(x)
            batch_reprs = []
            for sent in word_strs:
                word_reprs = self.char_embedder(sent, char_to_ix)
                batch_reprs.append(word_reprs)
            emb_char = nn.utils.rnn.pad_sequence(batch_reprs, batch_first=True)
            emb = torch.cat([emb_word, emb_char], dim=-1)
            emb = self.combine_proj(emb)
        elif self.repr_type == 'c':
            batch_reprs = []
            for sent, idx_seq in zip(word_strs, x):
                reprs = []
                for word, idx in zip(sent, idx_seq):
                    pre = word[:3] if len(word) >= 3 else word
                    suf = word[-3:] if len(word) >= 3 else word
                    pre_ix = prefix_to_ix.get(pre, prefix_to_ix['<UNK>'])
                    suf_ix = suffix_to_ix.get(suf, suffix_to_ix['<UNK>'])
                    word_emb = self.embedding(idx)
                    pre_emb = self.prefix_embedding(torch.tensor(pre_ix).to(word_emb.device))
                    suf_emb = self.suffix_embedding(torch.tensor(suf_ix).to(word_emb.device))
                    total = word_emb + pre_emb + suf_emb
                    reprs.append(total)
                batch_reprs.append(torch.stack(reprs))
            emb = nn.utils.rnn.pad_sequence(batch_reprs, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out1, _ = self.lstm1(packed)
        out1, _ = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        out2, _ = self.lstm2(nn.utils.rnn.pack_padded_sequence(out1, lengths, batch_first=True, enforce_sorted=False))
        out2, _ = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True)
        out2 = self.dropout(out2)
        logits = self.classifier(out2)
        return logits

def build_vocab(data_file):
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    tag_to_ix = {}
    char_to_ix = {'<PAD>': 0, '<UNK>': 1}
    prefix_to_ix = {'<UNK>': 0}
    suffix_to_ix = {'<UNK>': 0}
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                word, tag = line.split()
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
                if tag not in tag_to_ix:
                    tag_to_ix[tag] = len(tag_to_ix)
                for c in word:
                    if c not in char_to_ix:
                        char_to_ix[c] = len(char_to_ix)
                pre = word[:3] if len(word) >= 3 else word
                suf = word[-3:] if len(word) >= 3 else word
                if pre not in prefix_to_ix:
                    prefix_to_ix[pre] = len(prefix_to_ix)
                if suf not in suffix_to_ix:
                    suffix_to_ix[suf] = len(suffix_to_ix)
    return word_to_ix, tag_to_ix, char_to_ix, prefix_to_ix, suffix_to_ix
def evaluate(model, dataloader, device, char_to_ix=None, prefix_to_ix=None, suffix_to_ix=None):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets, lengths, word_strs in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, lengths, word_strs, char_to_ix, prefix_to_ix, suffix_to_ix)
            preds = torch.argmax(logits, dim=-1)
            mask = targets != -100
            correct += ((preds == targets) & mask).sum().item()
            total += mask.sum().item()
    return correct / total
import csv

def save_accuracies(dev_accuracies, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["num_sentences", "accuracy"])
        for step, acc in dev_accuracies:
            writer.writerow([step, acc])


def main():
    repr_type = sys.argv[1]
    train_file = sys.argv[2]
    model_file = sys.argv[3]
    dev_file = sys.argv[4]

    word_to_ix, tag_to_ix, char_to_ix, prefix_to_ix, suffix_to_ix = build_vocab(train_file)
    dataset = SequenceDataset(train_file, word_to_ix, tag_to_ix, repr_type)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)
    dev_dataset = SequenceDataset(dev_file, word_to_ix, tag_to_ix, repr_type)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, collate_fn=pad_collate)

    model = BiLSTMTagger(
        len(word_to_ix), len(tag_to_ix),
        len(char_to_ix), repr_type,
        prefix_size=len(prefix_to_ix),
        suffix_size=len(suffix_to_ix)
    )

    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    dev_accuracies = []
    for epoch in range(5):
        model.train()
        for inputs, targets, lengths, word_strs in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs, lengths, word_strs, char_to_ix, prefix_to_ix, suffix_to_ix)
            loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()

            step += len(word_strs)
            if step >= 500:
                acc = evaluate(model, dev_loader, device, char_to_ix, prefix_to_ix, suffix_to_ix)
                print(f"Dev accuracy after {step} examples: {acc:.4f}")
                dev_accuracies.append((step, acc))
                step = 0
    save_accuracies(dev_accuracies, model_file + ".accuracies.csv")

    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_ix': word_to_ix,
        'tag_to_ix': tag_to_ix,
        'char_to_ix': char_to_ix,
        'prefix_to_ix': prefix_to_ix,
        'suffix_to_ix': suffix_to_ix
    }, model_file)


if __name__ == '__main__':
    main()