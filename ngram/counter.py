from collections import defaultdict # توضیح در فایل tokenizer/build_vocab.py
from typing import List, Tuple, Dict # توضیح در فایل tokenizer/build_vocab.py

class NGramCounter: # این کلاسی است که که مسئول شمارش n-gram ها و نگهداری آمار احتمالات بعدی هر توکن است.
    def __init__(self, n: int): # تابع سازنده کلاس که یک n به عنوان ورودی می گیرد که تعیین می کند مدل Unigram, Bigram, Trigram, ... باشد.
        if n < 1: # چک می کند که n کوچک تر از 1 باشد.
            raise ValueError("n باید >= 1 باشد") # اگر این شرط برقرار بود خطا بده چون n حتما باید بزرگتر مساوی 1 باشد
        self.n = n # قرار دادن مقدار n

        # counts: { context_tuple : {next_token_id: count} }
        self.counts: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(lambda: defaultdict(int)) # این دیکشنری را ساختیم که بگیم تعداد تکرار آمدن یک کلمه بعد از context چقدر است و از این طریق می تونیم بیشترین احتمال برای کلمه بعدی یک context خاص را بیابیم.

    def update(self, token_ids: List[int]):
        """این تابع با استفاده از token_ids context ها و next_token را از روی این دنباله می سازد. و میشه گفت در این مرحله به تنظیم آمار و احتمالات
        برای کلمه بعدی در هر کانتکست می پردازیم."""

        if len(token_ids) < self.n: # اگر n بزرگتر از طول token_ids بود اون وقت یک return پوچ انجام بده چون این امکان پذیر نیست
            return
        
        for i in range(len(token_ids) - self.n + 1): # تا آخرین ایندکس ممکن را از token-ids پیمایش می کنیم
            context = tuple(token_ids[i:i + self.n - 1]) # n-1 توکن به عنوان context
            next_token = token_ids[i + self.n - 1] # توکن بعدی
            self.counts[context][next_token] += 1 # تعداد آمدن next_token بعد از context را یکی افزایش می دیم. اگر احیانا context یا next_token در counts نبود یک کانتکست خالی و next_token = 0 می سازیم.

    def get_next_counts(self, context: Tuple[int, ...]) -> Dict[int, int]:
        """اینجا ما تعداد کلمات آورده شده بعد از context را خروجی می دهیم"""
        if len(context) != self.n - 1: # باید حتما طول context برابر با n - 1 باشد وگرنه خطا می گیریم.
            raise ValueError(f"طول context باید {self.n - 1} باشد")
        return dict(self.counts.get(context, {})) # count ها را بر می گردونیم.
