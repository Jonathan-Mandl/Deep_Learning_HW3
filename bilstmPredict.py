import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class SequenceDatasetPredict:
    def __init__(self, data_file, word_to_ix):
        self.sentences = []
        self.words = []
        with open(data_file) as f:
            sentence = []
            word_list = []
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        self.sentences.append(sentence)
                        self.words.append(word_list)
                        sentence = []
                        word_list = []
                else:
                    sentence.append(word_to_ix.get(line, word_to_ix['<UNK>']))
                    word_list.append(line)
            if sentence:
                self.sentences.append(sentence)
                self.words.append(word_list)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), self.words[idx]

def pad_collate_predict(batch):
    inputs, word_strs = zip(*batch)
    lengths = [len(s) for s in inputs]
    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return padded_inputs, lengths, word_strs

class CharBiLSTMEmbedder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, char_hidden_dim):
        super(CharBiLSTMEmbedder, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        self.char_lstm = nn.LSTM(char_emb_dim, char_hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, word_list, char_to_ix):
        reps = []
        for word in word_list:
            char_ids = [char_to_ix.get(c, char_to_ix['<UNK>']) for c in word]
            char_tensor = torch.tensor(char_ids).unsqueeze(0)
            char_embs = self.char_embedding(char_tensor)
            _, (hn, _) = self.char_lstm(char_embs)
            rep = torch.cat((hn[0], hn[1]), dim=1)
            reps.append(rep.squeeze(0))
        return torch.stack(reps)

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, char_vocab_size=None, repr_type='a', emb_dim=30, hidden_dim=50, dropout=0.3, prefix_size=None, suffix_size=None):
        super(BiLSTMTagger, self).__init__()
        self.repr_type = repr_type
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)

        if repr_type in ['b', 'd']:
            self.char_embedder = CharBiLSTMEmbedder(char_vocab_size, 15, emb_dim // 2)

        if repr_type == 'd':
            self.combine_proj = nn.Linear(emb_dim + emb_dim, emb_dim)

        if repr_type == 'c':
            self.prefix_embedding = nn.Embedding(prefix_size, emb_dim)
            self.suffix_embedding = nn.Embedding(suffix_size, emb_dim)

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
            for sent in word_strs:
                reprs = []
                for word in sent:
                    word_ix = x[0][len(reprs)]
                    pre = word[:3] if len(word) >= 3 else word
                    suf = word[-3:] if len(word) >= 3 else word
                    pre_ix = prefix_to_ix.get(pre, prefix_to_ix['<UNK>'])
                    suf_ix = suffix_to_ix.get(suf, suffix_to_ix['<UNK>'])
                    word_emb = self.embedding(word_ix)
                    pre_emb = self.prefix_embedding(torch.tensor(pre_ix))
                    suf_emb = self.suffix_embedding(torch.tensor(suf_ix))
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

def main():
    repr_type = sys.argv[1]
    model_file = sys.argv[2]
    input_file = sys.argv[3]

    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    word_to_ix = checkpoint['word_to_ix']
    tag_to_ix = checkpoint['tag_to_ix']
    char_to_ix = checkpoint.get('char_to_ix', None)
    prefix_to_ix = checkpoint.get('prefix_to_ix', None)
    suffix_to_ix = checkpoint.get('suffix_to_ix', None)
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}

    dataset = SequenceDatasetPredict(input_file, word_to_ix)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_predict)

    model = BiLSTMTagger(
        len(word_to_ix), len(tag_to_ix),
        len(char_to_ix) if char_to_ix else None,
        repr_type,
        prefix_size=len(prefix_to_ix) if prefix_to_ix else None,
        suffix_size=len(suffix_to_ix) if suffix_to_ix else None
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        with open(input_file) as original:
            input_lines = original.readlines()
            sent_id = 0
            for inputs, lengths, word_strs in dataloader:
                logits = model(inputs, lengths, word_strs, char_to_ix, prefix_to_ix, suffix_to_ix)
                preds = torch.argmax(logits, dim=-1)[0][:lengths[0]]
                for token_id in preds:
                    while not input_lines[sent_id].strip():
                        print()
                        sent_id += 1
                    word = input_lines[sent_id].strip()
                    tag = ix_to_tag[token_id.item()]
                    print(f"{word} {tag}")
                    sent_id += 1
                print()
                sent_id += 1


if __name__ == '__main__':
    main()
