class Tokenizer:
    def __init__(self, special_tokens=["[PAD]", "[UNK]", "[EOS]"]):
        self.vocab = {tok: i for i, tok in enumerate(special_tokens)}

    def build_vocab(self, text):
        for char in set(text):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

    def encode(self, text):
        return [self.vocab.get(c, self.vocab["[UNK]"]) for c in text] + [self.vocab["[EOS]"]]

    def decode(self, tokens):
        inv = {i: tok for tok, i in self.vocab.items()}
        return "".join([inv.get(i, "?") for i in tokens])
