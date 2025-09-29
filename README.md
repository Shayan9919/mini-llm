#Mini Language Model (Educational)

This is a minimal from-scratch language model project.  
It includes a simple tokenizer and a 2-layer Transformer trained on a sample text data.
The aim was to understand the mechanics behind modern LMs at a small scale and experiment

##Features
- Custom tokenizer with reserved tokens ([PAD], [UNK], [EOS])
- PyTorch training loop with gradient clipping and learning-rate warmup
- Evaluation harness: loss curves and sample generations
- Unit test for tokenizer edge cases

##Why I Built This

During my self-study of Karpathy’s neural networks material, I wanted to go beyond theory and code a test GPT myself. The process taught me about tokenization edge cases, exploding gradients and how model design choices affect stability and training speed.
This project is not about performance, but about learning the building blocks of transformer-style models. It complements my work on retrieval-augmented generation by strengthening my understanding of embeddings, tokenization and training stability.