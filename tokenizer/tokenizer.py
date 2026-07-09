"""توکنایزر حرفه‌ای متن فارسی"""

import re # برای استفاده از الگو هایی که نوشتیم
from typing import List, FrozenSet, Optional # type hinting

from .cleaner import TextCleaner # پاکسازی ساختار های غیراستاندارد
from .punctuation_set import TEXT_SYMBOLS # تمام علائم نگارشی و نماد های موجود در UCD یا می تونیم بگیم در تمام جهان
from .patterns import (
    PERSIAN_PREFIX_PATTERN,
    PERSIAN_SUFFIX_PATTERN,
    REPEATED_CHAR_PATTERN,
    MULTI_WHITESPACE_PATTERN,
) # الگو ها


class PersianTokenizer:
    """
    توکنایزر حرفه‌ای متن فارسی با قابلیت‌ های:
    - جداسازی پیشوند های فعلی (می, نمی, بی)
    - جداسازی پسوندهای جمع و ملکی (ها, های, ام, ات و ...)
    - جداسازی علائم نگارشی
    - کاهش حروف تکراری
    """

    # Pre-compile الگوی علائم نگارشی یک‌ بار
    _PUNCT_PATTERN = re.compile(
        f"([{re.escape(''.join(TEXT_SYMBOLS))}])"
    )

    def __init__(
        self,
        separate_prefix_suffix: bool = True,
        separate_punctuations: bool = True,
        normalize_repeated: bool = True,
    ):
        self.separate_prefix_suffix = separate_prefix_suffix
        self.separate_punctuations = separate_punctuations
        self.normalize_repeated = normalize_repeated

    @staticmethod
    def normalize_repeated_func(match: re.Match[str]) -> str:
        """
        _این تابع قرار است توسط re.sub() صدا زده شود._\n
        _وقتی در re.sub به جای یک رشته, یک تابع بدهی: پایتون برای هر match این تابع را اجرا می‌ کند._

        Args:
            match (re.Match[str]): _تابع sub() هر بار که یک match پیدا کند, خودش یک شی از نوع match می‌ سازد و آن را به تابع تو می‌ دهد_

        Returns:
            str: _خروجی حاوی کلمه عادی است_
        """
        chars = match.group(0) # در واقع کل قسمت match شده را داخل متغیر chars می ریزد
        if chars[0].isascii(): # با توجه به حرف اول در اصل تعیین می کند که کلمه انگلیسی است یا خیر
            # چرا chars[0] : چون regex ما تضمین کرده که تمام حروف یکسان اند
            return chars[:2]   # اگر انگلیسی باشد دو حرف رو نگه می داریم
        return chars[:1]       # ما تو زبان فارسی استاندارد کلمه ای نداریم که حروف پشت سرهم آن از یک نوع باشند پس به یک حرف کاهش می دهیم

    def tokenize(self, text: str) -> List[str]:
        """تبدیل متن به لیست توکن‌ ها"""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        if not text.strip(): # در واقع برای بررسی متن های خالی ولی با فاصله است
            return [] # یک لیست خالی برمی گردانیم که نشان دهنده این است که هیچ توکنی پیدا نشد

        # مرحله 1: پاکسازی
        text = TextCleaner.clean(text)

        # مرحله 2: کاهش حروف تکراری
        if self.normalize_repeated:
            text = REPEATED_CHAR_PATTERN.sub(self.normalize_repeated_func, text)

        # مرحله 3: جداسازی پیشوند و پسوند فارسی
        if self.separate_prefix_suffix:
            text = PERSIAN_PREFIX_PATTERN.sub(r"\1 ", text)
            text = PERSIAN_SUFFIX_PATTERN.sub(r" \1", text)

        # مرحله 4: جداسازی علائم نگارشی
        if self.separate_punctuations:
            text = self._PUNCT_PATTERN.sub(r" \1 ", text)

        # مرحله 5: فاصله‌ های نهایی
        text = MULTI_WHITESPACE_PATTERN.sub(" ", text).strip()

        # مرحله 6: توکنایز نهایی
        return [token for token in text.split(" ") if token]