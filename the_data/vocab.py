class CharVocab:
    def __init__(self, chars, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.specials = specials
        self.itos = specials + sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def text2ids(self, text):
        return [self.stoi.get(ch, self.stoi["<unk>"]) for ch in text]

    def ids2text(self, ids):
        return ''.join([self.itos[i] for i in ids if i not in [self.stoi["<pad>"], self.stoi["<sos>"], self.stoi["<eos>"]]])
