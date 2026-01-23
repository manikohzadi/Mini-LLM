# rnn_model.py (Built-in only, LSTM industrial)
import math
import random
from typing import List
from rnn.rnn_cell import LSTMCell  # سلول حرفه‌ای LSTM که توضیح دادی

class LSTMModel:
    def init(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.2):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Embedding trainable
        self.embedding = [[(random.random() - 0.5) * 0.02 for _ in range(embedding_dim)] for _ in range(vocab_size)]

        # LSTM cell حرفه‌ای
        self.lstm_cell = LSTMCell(embedding_dim, hidden_dim)

        # Fully connected layer برای تولید logits
        self.W_out = [[(random.random() - 0.5) * 0.02 for _ in range(vocab_size)] for _ in range(hidden_dim)]
        self.b_out = [0.0 for _ in range(vocab_size)]

    def forward(self, input_ids: List[int]):
        """یک پاس جلو با تمام توکن‌ها"""
        outputs = []

        for t in input_ids:
            x_t = self.embedding[t]
            h_t = self.lstm_cell.forward(x_t)  # فقط ورودی x را می‌دهیم

            # Dropout در hidden state
            if self.dropout > 0:
                mask = [1.0 if random.random() >= self.dropout else 0.0 for _ in h_t]
                h_t = [hi * m / (1.0 - self.dropout) for hi, m in zip(h_t, mask)]

            # Fully connected -> logits
            logits = [sum(h_t[i] * self.W_out[i][j] for i in range(self.hidden_dim)) + self.b_out[j] for j in range(self.vocab_size)]
            probs = self.softmax(logits)
            outputs.append(probs)

        return outputs

    @staticmethod
    def softmax(logits: List[float]) -> List[float]:
        """Softmax ایمن از overflow"""
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def generate(self, start_ids: List[int], max_length: int = 50, temperature: float = 1.0, top_k: int = 40):
        """تولید متن با LSTM"""
        output_ids = start_ids[:]

        for _ in range(max_length):
            probs_seq = self.forward([output_ids[-1]])
            probs = probs_seq[-1]

            # Temperature
            adjusted_probs = [p ** (1.0 / temperature) for p in probs]
            s = sum(adjusted_probs)
            adjusted_probs = [p / s for p in adjusted_probs]

            # Top-k
            top_indices = sorted(range(len(adjusted_probs)), key=lambda i: adjusted_probs[i], reverse=True)[:top_k]
            top_probs = [adjusted_probs[i] for i in top_indices]
            s = sum(top_probs)
            top_probs = [p / s for p in top_probs]

            next_token = random.choices(top_indices, weights=top_probs)[0]
            output_ids.append(next_token)

        return output_ids