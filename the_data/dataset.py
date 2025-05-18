import torch
from torch.utils.data import Dataset


class TransliterationDataset(Dataset):
    def __init__(self, tsv_path, input_vocab, output_vocab):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.pairs = []

        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                tgt, src, freq = line.strip().split('\t')
                src_ids = input_vocab.text2ids(src)
                tgt_ids = [output_vocab.stoi["<sos>"]] + output_vocab.text2ids(tgt) + [output_vocab.stoi["<eos>"]]
                self.pairs.append((
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


