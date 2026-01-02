import json # اضافه کردن کتابخانه JavaScript Object Notation برای ذخیره vocab در قالب json برای کار کردن با API ها(در آبنده)
from collections import Counter # Counter مثل یک دیکشنری است که داخل آن کلید ها خود عنصر و مقدار ها تعداد تکرار آن عنصر اند که ورودی این class یک Iterable است
from typing import List, Dict # افزودن این کتابخانه برای type annotations توابع و بالا بردن readability کد

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] # توضیح هر توکن مخصوص:
# <PAD> برای وقتی است که می خواهیم دو جمله را تراز کنیم مثل hello <PAD> و hello world
# <UNK> برای وقتی است که یک توکن در vocab ناشناخته است مثل hello gfhfghgbgh تبدیل میشه به hello <UNK>
# <BOS> برای وقتی است که می خواهیم اول یا شروع جمله را مشخص کنیم مثل <BOS> من مانی کهزادی هستم
# <EOS> برای وقتی است که می خواهیم انتها یا پایان جمله را مشخص کنیم مثل من مانی کهزادی هست <EOS>
# مثال کامل:
# hello world nhtvhbrehgfd mani kohzadi
# hello world mani
# تبدیل میشه به:
# <BOS> hello world <UNK> mani kohzadi <EOS>
# <BOS> hello world mani <PAD> <PAD> <EOS>


def build_vocab(all_tokens: List[List[str]], max_vocab_size: int = 100000) -> Dict[str, Dict]:
    """این تابع یک واژگان از لیستی از لیست توکن ها می سازد و هم خود واژگان را در قالب دیکشنری خروجی می دهد هم آن را در قالب فایل JSON ذخیره می کند"""

    flat_tokens = [tok for doc in all_tokens for tok in doc] # تمام توکن ها را فقط به یک لیست صاف تبدیل می کنهد
    
    counter = Counter(flat_tokens) # ساخت یک Counter از تمامی توکن ها برای بدست آوردن شمارش هر توکن
    
    most_common = counter.most_common(max_vocab_size) # پیدا کردن max_vocab_size تا توکن پر تکرار
    tokens = SPECIAL_TOKENS + [tok for tok, _ in most_common if tok not in SPECIAL_TOKENS] # اضافه کردن توکن های اختصاصی به اول vocab و بعد اضافه کردن بقیه توکن ها
    
    # ساخت mapping دوطرفه
    token_to_id = {tok: idx for idx, tok in enumerate(tokens)} # توکن در مقابل id
    id_to_token = {idx: tok for tok, idx in token_to_id.items()} # id در مقابل توکن
    
    vocab = {"token_to_id": token_to_id, "id_to_token": id_to_token} # ساخت واژگان نهایی
    
    # ذخیره واژگان در فایل JSON
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    
    return vocab # برگرداندن خود vocab
