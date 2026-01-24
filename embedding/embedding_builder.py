import json # اضافه کردن این کتابخانه برای ذخیره سازی فایل word embeddings به صورت json
import math # اضافه کردن این کتابخانه برای پیاده سازی کردن L2 Normalize
from collections import defaultdict # اضافه کردن این ساختمان داده برای یافتن تعداد تکرار یک کلمه بعد از کانتکست
from typing import List, Dict # اضافه کردن این کتابخانه برای مشخص کردن نوع ورودی و خروجی برای بالا بردن خوانایی و دیباگینگ راحت تر
import hashlib
import struct


class DistributionalEmbeddingBuilder: # این کلاس برای ساخت word embeddings بر پایه فرضیه توزیعی
    def __init__(
        self,
        vocab: Dict[str, Dict], # واژگان دو طرفه
        embedding_dim: int = 128, # طول هر word_embedding
        window_size: int = 2 # طول پنجره یا همون کانتکست
    ):
        self.token_to_id = vocab["token_to_id"] # دیکشنری توکن به شناسه
        self.id_to_token = vocab["id_to_token"] # دیکشنری شناسه به توکن

        self.embedding_dim = embedding_dim
        self.window_size = window_size

        # embedding[token_id] = [float, float, ...]
        self.embeddings = defaultdict( # یک دیکشنری که وقتی یک شناسه توکن را بهش میدی که در لیست کلید ها وجود نداره اونو رو با مقدار لیستی از 0.0 ها به طول ابعاد embedding ذخیره می کنه
            lambda: [0.0] * self.embedding_dim
        )

    def _hash_context(self, token: str) -> int:
        """هش پایدار، سریع و صنعتی برای نگاشت کانتکست به فضای embedding."""

        token = token.strip().lower() # نرمال سازی برای پایداری بیشتر

        # 2) هش cryptographic سریع و پایدار
        digest = hashlib.blake2b(
            token.encode("utf-8"),
            digest_size=8  # 64-bit
        ).digest()

        # 3) تبدیل بایت → عدد صحیح
        value = struct.unpack(">Q", digest)[0]

        # 4) فشرده‌سازی به فضای embedding
        return value % self.embedding_dim

    def update_from_sentence(self, tokens: List[str]):
        length = len(tokens)

        for i, center in enumerate(tokens):
            if center not in self.token_to_id:
                continue

            center_id = self.token_to_id[center]

            start = max(0, i - self.window_size)
            end = min(length, i + self.window_size + 1)

            for j in range(start, end):
                if j == i:
                    continue

                context_token = tokens[j]
                idx = self._hash_context(context_token)
                self.embeddings[center_id][idx] += 1.0

    def normalize(self):
        """
        log-scaling + L2 normalization
        """
        for token_id, vec in self.embeddings.items():
            # log scaling
            for i in range(len(vec)):
                vec[i] = math.log(1.0 + vec[i])

            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                for i in range(len(vec)):
                    vec[i] /= norm

    def save(self, path: str):
        """
        ذخیره embedding به صورت JSON
        """
        output = {
            "embedding_dim": self.embedding_dim,
            "embeddings": {
                str(token_id): vec
                for token_id, vec in self.embeddings.items()
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data