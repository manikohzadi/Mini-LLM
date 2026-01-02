import random # اضافه کردن این کتابخانه برای تولید اعداد تصادفی
from typing import Dict, List, Tuple # اضافه کردن این کتابخانه برای مشخص کردن نوع ورودی و خروجی توابع
import math # اضافه کردن این کتابخانه برای کار های ریاضی مثل exp برای softmax


def softmax(logits: Dict[int, float]) -> Dict[int, float]:
    if not logits: # بررسی کردن اینکه نمره ها خالی باشند
        return {} # یک دیکشنری خالی بر می گردانیم چون اگر این کار را نکنیم در خط 10 کد کرش می کند

    max_logit = max(logits.values()) # پیدا کردن بیشترین نمره

    exp_values = {} # دیکشنری نمرات که کلید آن توکن و مقدار آن نمره است
    for token, value in logits.items(): # حلقه زدن روی توکن و نمره آن
        exp_values[token] = math.exp(value - max_logit) # محاسبه کردن تابع نمایی با ورودی نمره کنونی منهای بیشترین نمره برای جلوگیری از overflow عددی

    total = sum(exp_values.values()) # محاسبه کردن مجموع نمره ها
    if total == 0: # بررسی کردن اینکه مجموع صفر شود به هر دلیلی مثل نمره های منفی خیلی بزرگ یا -inf
        return {token: 1 / len(exp_values) for token in exp_values} # اگر همچین اتفاقی افتاده بود حداقل یک توزیع یکنواخت می سازیم که مجموع بشه 1

    return {token: val / total for token, val in exp_values.items()} # اگر شرط بالا برقرار نبود با فرمول divide by sum توزیع یکنواخت می سازیم


def sample_with_temperature(probs: Dict[int, float], temperature: float = 1.0) -> int: # تابع sampler با روش temperature
    """نمونه‌گیری با Temperature"""
    if temperature <= 0: # بررسی کردن منفی یا صفر بودن temperature
        raise ValueError("Temperature must be > 0") # نمایش خطا زیرا که اگر جلوی این کار را نگیریم ممکن است منجر به تقسیم بر صفر یا رفتار غیرقابل تعریف شود

    # تغییر توزیع بر اساس temperature
    adjusted_probs = {token: prob ** (1/temperature) for token, prob in probs.items()} # احتمال هر توکن را به توان 1 تقسیم بر temperature می رسانیم
    # رفتار:

    #     temperature > 1 = توزیع یکنواخت‌تر (خلاق‌تر)

    #     temperature < 1 = توزیع تیزتر (محافظه‌کارتر)

    #     temperature = 1 = بدون تغییر

    # normalize
    adjusted_probs = softmax(adjusted_probs) # نرمال سازی احتمالات چون اگر این کار را نکنیم مجموع احتمالات 1 نمی شود که این درست نیست

    # نمونه‌گیری
    r = random.random() # توضیحات در فایل model.py کلاس NGramLanguageModel و متد sample_next
    cumulative = 0.0
    for token, prob in adjusted_probs.items():
        cumulative += prob
        if r <= cumulative:
            return token
    return random.choice(list(adjusted_probs.keys()))


def top_k_sampling(probs: Dict[int, float], k: int = 10) -> int:
    """نمونه‌گیری Top-k"""
    if k <= 0: # چک می کنیم که صفر یا منفی نباشه
        raise ValueError("k must be > 0") # اگر بود باید خطا بدیم که جلوی رفتار های تعریف نشده و خطرناک را بگیریم.

    # مرتب‌سازی و گرفتن k توکن با بیشترین احتمال
    top_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k] # توکن ها را از بزرگ به کوچک با توجه به احتمال آنها مرتب می کنیم و k تای اول را می گیریم
    normalized_probs = softmax(dict(top_tokens)) # احتمال ها را نرمال سازی می کنیم که مجموع آنها 1 شود

    # نمونه‌گیری
    r = random.random() # توضیحات در فایل model.py کلاس NGramLanguageModel و متد sample_next
    cumulative = 0.0
    for token, prob in normalized_probs.items():
        cumulative += prob
        if r <= cumulative:
            return token
    return random.choice(list(normalized_probs.keys()))


def top_p_sampling(probs: Dict[int, float], p: float = 0.9) -> int:
    """نمونه‌گیری Top-p (Nucleus sampling)"""
    if not (0 < p <= 1.0): # بررسی کردن اینکه p در بازه 0 تا 1 نباشد
        raise ValueError("p must be in (0, 1]") # اگر شرط بالا برقرار بود خطا می دیم چون p یک احتمال است و باید بین 0 و 1 باشد

    # مرتب‌سازی توکن‌ها بر اساس احتمال
    sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True) # مرتب کردم توکن ها بر اساس احتمال آنها از بزرگ به کوچک

    cumulative = 0.0 # جمع تجمعی
    top_tokens = [] # توکن هایی که جمع آنها بزرگ تر یا مساوی p است. یعنی دور و بر p است. و این توکن ها بیشترین احتمال را دارند
    for token, prob in sorted_tokens: # حلقه زدن روی توکن های مرتب شده از کوچک به بزرگ
        cumulative += prob # اضافه کردن احتمال این توکن به جمع تجمعی 
        top_tokens.append((token, prob)) # اضافه کردن این توکن به top_tokens
        if cumulative >= p: # اگر جمع تجمعی  بیشتر از p بود
            break # حلقه را متوقف می کنیم

    # normalize
    normalized_probs = softmax(dict(top_tokens)) # نرمال سازی احتمال های top_tokens طوری که توکن هایی با احتمال بالا تر غالب شوند و در نهایت مجموع 1 شود

    # نمونه‌گیری
    r = random.random() # توضیحات مثل قبل 
    cumulative = 0.0
    for token, prob in normalized_probs.items():
        cumulative += prob
        if r <= cumulative:
            return token
    return random.choice(list(normalized_probs.keys()))
