"""API عمومی ماژول"""

from typing import List, Optional # type hinting
from .tokenizer import PersianTokenizer # توکنایزر حرفه ای فارسی
from .cleaner import TextCleaner # مدیریت ساختار های غیراستاندارد متن
from .normalizer import PersianNormalizer # نرمال سازی یونیکد
# . : Relative Import یعنی از همین package

# Singleton برای استفاده راحت
_default_tokenizer: Optional[PersianTokenizer] = None # اول یک متغیر سراسری ساخته می‌ شود.
#   1. اسمش _default_tokenizer است که _ به این معنی است که برنامه نویسان دیگر حق استفاده از این متغیر رو بیرون از این فایل ندارند. و این فقط یک قرارداد نانوشته بین برنامه نویسان است.
#   2. فعلا هیچ شیئ وجود ندارد, پس None داخلش قرار می دهیم.
#   3. فرض کن ساختن Tokenizer خیلی زمان‌ بر باشد. اگر هر بار tokenize() را صدا بزنی یک شیء جدید ساخته شود خیلی کند می‌ شود.
#   4. پس یک بار شیء ساخته می‌ شود. بعد همیشه همان استفاده می‌ شود. به این می‌ گویند Singleton Pattern


def get_tokenizer() -> PersianTokenizer:
    """دریافت نمونه پیش‌ فرض توکنایزر"""
    global _default_tokenizer # این خط یعنی آن متغیر سراسری را استفاده کن.
    # اگر این را ننویسیم پایتون فکر می‌ کند _default_tokenizer یک متغیر محلی است.
    # و این خط _default_tokenizer = PersianTokenizer() خطا می‌ دهد.

    if _default_tokenizer is None: # یعنی آیا Tokenizer قبلا ساخته نشده
        _default_tokenizer = PersianTokenizer() # اینجا Constructor صدا زده می‌ شود.
    return _default_tokenizer # اگر توکنایزر قبلا ساخته شده بود دیگر آن را نمی سازیم و همان شئ قبلی را برمی گردانیم


def tokenize(text: str) -> List[str]:
    """تابع سریع برای توکنایز متن"""
    # این یک Wrapper است.
    # یعنی به جای اینکه کاربر بنویسد tokenizer = get_tokenizer(); tokens = tokenizer.tokenize(text) فقط می‌ نویسد tokenize(text)

    return get_tokenizer().tokenize(text)


def clean_text(text: str) -> str:
    """تابع سریع برای پاکسازی متن"""
    # باز هم Wrapper است.

    return TextCleaner.clean(text) # شاید بگویید چرا از Singleton Pattern استفاده نکردیم؟ چون clean یک staticmethod است و اصلا نیازی به ساختن شئ ندارد.


__all__ = [
    "PersianTokenizer",
    "TextCleaner",
    "PersianNormalizer",
    "tokenize",
    "clean_text",
    "get_tokenizer",
]

# اگر کسی بنویسد from tokenizer import * فقط چیز هایی export می شوند که اسمشان در لیست __all__ باشد.
# اگر __all__ نباشد تقریبا همه چیز export می شود.