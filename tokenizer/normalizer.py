"""نرمال‌ سازی یونیکد فارسی/عربی"""

from .constants import (
    ARABIC_TO_PERSIAN_MAP,
    DIGITS_SOURCE,
    DIGITS_TARGET,
) # ثابت ها

# ساخت جدول ترجمه یک‌ بار در زمان بارگذاری ماژول
_TRANSLATION_TABLE: dict = str.maketrans(
    # این تابع یک Translation Table می‌ سازد
    # Translation Table یعنی یک دیکشنری مخصوص که به str.translate() میگه:
    #   هر وقت فلان کاراکتر رو دیدی, اون رو با این کاراکتر جایگزین کن
    "".join(ARABIC_TO_PERSIAN_MAP.keys()) + DIGITS_SOURCE,
    "".join(ARABIC_TO_PERSIAN_MAP.values()) + DIGITS_TARGET,
)
# داخل Translation Table کلید ها کد یونیکد هستند، نه خود کاراکتر. و برای این کار هم از ord() استفاده می کنیم

class PersianNormalizer:
    """نرمال‌ سازی یونیکد فارسی/عربی"""

    @staticmethod
    def normalize(text: str) -> str:
        """
        تبدیل کاراکتر های عربی به فارسی و اعداد فارسی/عربی به انگلیسی.

        مثال:
            >>> PersianNormalizer.normalize("مدرسة ١٢٣")
            'مدرسه 123'
        """
        if not text: # اگر متن خالی بود یا None بود
            return text # آنگاه خود متن که یا خالی است و یا None است را برمی گرداند
        return text.translate(_TRANSLATION_TABLE) # تبدیل کاراکتر و اعداد عربی به فارسی