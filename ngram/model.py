from typing import Dict, Tuple # توضیحات در فایل .counter/NGramCounter
from .counter import NGramCounter # اضافه کردن شمارنده n-gram
from .sampler import sample_with_temperature, top_k_sampling, top_p_sampling



class NGramLanguageModel: # کلاس مخصوص مدل زبانی n-gram
    def __init__(self, counter: NGramCounter, vocab_size: int):
        self.counter = counter # گرفتن شی counter
        self.vocab_size = vocab_size # گرفتن طول واژگان
        self.n = counter.n # گرفتن n از شی counter

    def get_next_probs(self, context: Tuple[int, ...]) -> Dict[int, float]: # این تابع تعداد را تبدیل به یه احتمال می کند با استفاده از Add-One Smoothing
        """احتمال هر توکن بعدی را با smoothing محاسبه می‌کند"""
        counts = self.counter.get_next_counts(context) # گرفتن تعداد توکن های بعدی آمده بعد از context با شی counter

        total_count = sum(counts.values()) # جمع کل count ها
        vocab_size = self.vocab_size # طول vocab

        # Add-One Smoothing
        probs = {} # دیکشنری احتمالات
        for token_id in range(vocab_size): # پیمایش کل توکن ها
            count = counts.get(token_id, 0) # گرفتن count آن توکن اگر نبود صفر
            probs[token_id] = (count + 1) / (total_count + vocab_size) # ذخیره احتمال آن توکن با فرمول Add-One Smoothing

        return probs # برگرداندن احتمالات رخ دادن توکن بعدی

    def sample_next(
        self,
        context,
        strategy="top_p",
        temperature=1.0,
        top_k=40,
        top_p=0.9
    ):
        probs = self.next_token_probs(context)

        if strategy == "temperature":
            return sample_with_temperature(probs, temperature)

        elif strategy == "top_k":
            return top_k_sampling(probs, top_k)

        elif strategy == "top_p":
            return top_p_sampling(probs, top_p)

        else:
            raise ValueError("Unknown sampling strategy")
