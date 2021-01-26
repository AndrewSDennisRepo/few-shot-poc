import torch.nn as nn
import torch.nn.functional as F




class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):

    	super().__init__()

    	self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
    	x = self.embedding(inputs)
    	return x
