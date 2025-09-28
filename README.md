# Mini Language Model (Educational)

This is a minimal from-scratch language model project.  
It includes a simple tokenizer and a 2-layer Transformer trained on toy text data.

## Features
- Custom tokenizer with reserved tokens ([PAD], [UNK], [EOS])
- PyTorch training loop with gradient clipping and learning-rate warmup
- Evaluation harness: loss curves and sample generations
- Unit test for tokenizer edge cases