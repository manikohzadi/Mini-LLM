from typing import List, Tuple # برای type annotations
from ngram.model import NGramLanguageModel # مدل زبانی n-gram

class TextGenerator: # کلاس مخصوص تولید متن با استفاده از n-gram
    def __init__(self, lm: NGramLanguageModel, id_to_token: dict, token_to_id: dict):
        """
        lm: شی NGramLanguageModel
        id_to_token: دیکشنری {id: token}
        token_to_id: دیکشنری {token: id}
        """
        self.lm = lm # شی مدل زبانی n-gram
        self.id_to_token = id_to_token # واژگانی که کلید آن id و مقدار آن token است
        self.token_to_id = token_to_id # برعکس واژگان بالا

        self.BOS = token_to_id["<BOS>"] # id توکن ویژه BOS که نشان دهنده شروع جمله است
        self.EOS = token_to_id["<EOS>"] # id توکن ویژه EOS که نشان دهنده پایان جمله است

    def generate(self, max_length: int = 50, strategy="top_p", temperature=1.0, top_k=40, top_p_val=0.9) -> List[str]:
        """تولید متن بدون پرامپت"""
        n = self.lm.n # گرفتن n از طریق مدل زبانی n-gram
        context = [self.BOS] * (n - 1) # درست کردن contextی از id BOS به طول n - 1. این برای شبیه سازی یک context خالی است.
        output_ids = [] # id توکن ها خروجی

        for _ in range(max_length): # تکرار فرایند زیر به اندازه max_length بار
            context_tuple = tuple(context) # تبدیل نوع context از list به tuple
            next_id = self.lm.sample_next(
                context_tuple,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p_val
            ) # انتخاب توکن بعدی

            if next_id == self.EOS: # اگر احیانا مدل گفت که این پایان جمله است
                break # حلقه را قطع می کنیم

            output_ids.append(next_id) # اضافه کردن id توکن جدید
            context = context[1:] + [next_id] # اضافه کردن توکن کنونی تولید شده به آخر context و حذف توکن اول context

        return [self.id_to_token.get(tok_id, "<UNK>") for tok_id in output_ids] # تبدیل id ها به توکن های واقعی و تولید متن نهایی

    def generate_from_prompt(self, prompt: str, max_length: int = 50, strategy="top_p", temperature=1.0, top_k=40, top_p_val=0.9) -> str:
        """تولید متن بر اساس پرامپت"""
        from tokenizer.tokenizer import tokenize # اضافه کردن توکنایزر فارسی

        prompt_tokens = [self.token_to_id.get(tok, self.token_to_id["<UNK>"]) for tok in tokenize(prompt)] # تبدیل prompt به توکن ها
        n = self.lm.n # گرفتن n از طریق مدل زبانی n-gram
        # اگر طول prompt کمتر از n-1 باشد، آن را با <BOS> پر می‌کنیم
        context = [self.BOS] * max(0, n - 1 - len(prompt_tokens)) + prompt_tokens[-(n-1):] #  ساخت context با شرایط زیر:
        #   اگر طول context == n - 1: خود context را می گذاریم بمونه
        #   اگر طول context > n - 1: آخرین n-1 تا توکن از prompt را بردار
        #   اگر طول context < n - 1: قبل از context BOS اضافه می کنیم
        output_ids = list(context) # ساخت لیست خروجی نهایی که context هم شاملش می شود

        for _ in range(max_length):
            context_tuple = tuple(output_ids[-(n-1):]) # استفاده کردن از n - 1 آخرین توکن در خروجی ساخته شده به عنوان context
            next_id = self.lm.sample_next(
                context_tuple,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p_val
            ) # پیشبینی کردن توکن بعدی
            if next_id == self.EOS: # بررسی کردن اینکه مدل بگوید این پایان جمله است
                break # توقف می کنیم چون جمله تموم شده و مدل متن نهایی را آماده کره
            output_ids.append(next_id) # اگر شرط بالا برقرار نبود توکن تولید شده جدید را به متن نهایی اضافه می کنیم

        # تبدیل به توکن و حذف <BOS> و <EOS>
        generated_tokens = [
            self.id_to_token.get(tok_id, "<UNK>")
            for tok_id in output_ids
            if tok_id not in [self.BOS, self.EOS]
        ]
        return " ".join(generated_tokens) # ساخت متن نهایی
