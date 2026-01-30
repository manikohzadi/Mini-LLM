import random
import math
from collections import defaultdict
from typing import List, Dict


def sigmoid(x: float) -> float: # تابع فعال سازی sigmoid که ورودی را بین 0 و 1 نگه می دارد
    """تابع فعال سازی sigmoid نسخه پایدار عددی"""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


class Word2Vec:
    def init(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        negative_samples: int = 5,
        lr: float = 0.025
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.lr = lr

        # W_in و W_out
        self.W_in = {
            i: [random.uniform(-0.5, 0.5) / embedding_dim for _ in range(embedding_dim)]
            for i in range(vocab_size)
        }
        self.W_out = {
            i: [0.0 for _ in range(embedding_dim)]
            for i in range(vocab_size)
        }

    def _dot(self, v1, v2):
        return sum(a * b for a, b in zip(v1, v2))

    def _update(self, center, target, label):
        v_c = self.W_in[center]
        v_o = self.W_out[target]

        score = self._dot(v_c, v_o)
        pred = sigmoid(score)
        error = label - pred

        # گرادیان
        for i in range(self.embedding_dim):
            grad = self.lr * error
            temp = v_c[i]

            v_c[i] += grad * v_o[i]
            v_o[i] += grad * temp

    def train_pair(self, center: int, context: int):
        # positive
        self._update(center, context, 1)

        # negative sampling
        for _ in range(self.negative_samples):
            neg = random.randint(0, self.vocab_size - 1)
            if neg == context:
                continue
            self._update(center, neg, 0)

    def get_embedding(self, token_id: int) -> List[float]:
        return self.W_in[token_id]