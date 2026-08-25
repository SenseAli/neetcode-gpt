import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        embed_dim = 16
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        self.linear = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embedded = self.embedding(x)
        averaged = embedded.mean(dim=1)
        logits = self.linear(averaged)
        output = self.sigmoid(logits)
        return output
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        pass
