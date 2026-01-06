from typing import List
from rnn_cell import LSTMCell
import random
import math


class RNNLanguageModel:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Embedding matrix
        self.embeddings = [
            [random.uniform(-0.1, 0.1) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]

        # LSTM Cell
        self.lstm = LSTMCell(embedding_dim, hidden_size)

        # Output projection
        self.W_out = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
                      for _ in range(vocab_size)]
        self.b_out = [0.0 for _ in range(vocab_size)]

    def embed(self, token_id: int) -> List[float]:
        return self.embeddings[token_id]

    def softmax(self, logits: List[float]) -> List[float]:
        max_logit = max(logits)
        exp_vals = [math.exp(x - max_logit) for x in logits]
        s = sum(exp_vals)
        return [v / s for v in exp_vals]

    def forward_step(self, token_id: int) -> List[float]:
        """یک گام زمانی"""
        x = self.embed(token_id)
        h = self.lstm.forward(x)
        logits = [
            sum(w * h_i for w, h_i in zip(w_row, h)) + b
            for w_row, b in zip(self.W_out, self.b_out)
        ]
        return logits

    def predict_next(self, token_id: int, temperature: float = 1.0) -> int:
        logits = self.forward_step(token_id)

        # temperature
        scaled = [x / temperature for x in logits]
        probs = self.softmax(scaled)

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i

        return len(probs) - 1
