from the_data.vocab import CharVocab
from the_data.dataset import TransliterationDataset
from the_data.collate import collate_fn
from torch.utils.data import DataLoader
import torch 

def build_vocab(file_path, is_input=True):
    chars = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tgt, src, _ = line.strip().split('\t')
            text = src if is_input else tgt
            chars.update(text)
    return CharVocab(chars)

def get_dataloaders(train_path, dev_path, test_path, batch_size=32,pin_memory=True):
    pin_memory = torch.cuda.is_available()

    input_vocab = build_vocab(train_path, is_input=True)
    output_vocab = build_vocab(train_path, is_input=False)

    train_ds = TransliterationDataset(train_path, input_vocab, output_vocab)
    dev_ds = TransliterationDataset(dev_path, input_vocab, output_vocab)
    test_ds = TransliterationDataset(test_path, input_vocab, output_vocab)
    num_workers = 4 if torch.cuda.is_available() else 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_memory,
                          num_workers=num_workers,persistent_workers=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory,
                          num_workers=num_workers,persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory,
                          num_workers=num_workers,persistent_workers=True)

    return train_loader, dev_loader, test_loader, input_vocab, output_vocab

