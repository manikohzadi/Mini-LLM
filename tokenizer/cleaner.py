"""پاکسازی متن خام"""

from .normalizer import PersianNormalizer # نرمال ساز متن مخصوص فارسی
from .patterns import (
    URL_PATTERN,
    EMAIL_PATTERN,
    TAG_PATTERN,
    EMOJI_PATTERN,
    CONTROL_CHARS_PATTERN,
    MULTI_WHITESPACE_PATTERN,
) # الگو ها


class TextCleaner:
    """پاکسازی ساختار های ناخواسته از متن"""

    @staticmethod # متودی که نیازی به self(instance یا نمونه) یا cls(اطلاعات خود class) ندارد
    def clean(text: str) -> str:
        """
        پاکسازی کامل متن:
        1. حذف HTML, URL, Email, Emoji
        2. نرمال‌ سازی یونیکد فارسی
        3. تبدیل نیم‌ فاصله (ZWNJ) به فاصله
        4. حذف کاراکتر های کنترلی
        5. فاصله‌ های چندگانه تبدیل می شود به تک ‌فاصله

        مثال:
            >>> TextCleaner.clean("سلام   <b>دنیا</b>!")
            'سلام دنیا !'
        """
        if not isinstance(text, str): # اگر متن از جنس رشته نباشد
            raise TypeError(f"Expected str, got {type(text).__name__}") # آنگاه یک TypeError با متن مقابل به کاربر نشون بده
            # type(text) یک شی از نوع تایپ و یک property __name__ با مقدار اسم نوع text می باشد
        if not text: # اگر داخل متن خالی بود یا None بود
            return "" # آنگاه تو هم یک متن خالی به صورت مستقیم و بدون هیچ پردازشی بده

        # 1. حذف ساختار های خاص (ترتیب مهم است: HTML قبل از URL)
        text = TAG_PATTERN.sub(" ", text)
        text = URL_PATTERN.sub(" ", text)
        text = EMAIL_PATTERN.sub(" ", text)
        text = EMOJI_PATTERN.sub(" ", text)

        # 2. نرمال‌ سازی یونیکد
        text = PersianNormalizer.normalize(text)

        # 3. تبدیل نیم‌ فاصله به فاصله (برای جدا سازی صحیح کلمات)
        text = text.replace("\u200c", " ")

        # 4. حذف کاراکترهای کنترلی و BOM
        text = CONTROL_CHARS_PATTERN.sub("", text)

        # 5. فاصله‌ های چندگانه تبدیل می شود به تک‌ فاصله
        text = MULTI_WHITESPACE_PATTERN.sub(" ", text)

        return text.strip() # فاصله های ابتدایی و انتهایی رو پاک می کنه