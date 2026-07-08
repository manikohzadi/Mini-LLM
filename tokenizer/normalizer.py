"""نرمال‌ سازی یونیکد فارسی/عربی"""

from .constants import (
    ARABIC_TO_PERSIAN_MAP,
    DIGITS_SOURCE,
    DIGITS_TARGET,
) # ثابت ها

# ساخت جدول ترجمه یک‌ بار در زمان بارگذاری ماژول
_TRANSLATION_TABLE = str.maketrans({
    # این تابع یک Translation Table می‌ سازد
    # Translation Table یعنی یک دیکشنری مخصوص که به str.translate() میگه:
    #   هر وقت فلان کاراکتر رو دیدی, اون رو با این کاراکتر جایگزین کن
    **ARABIC_TO_PERSIAN_MAP,
    **dict(zip(DIGITS_SOURCE, DIGITS_TARGET)),
    # zip() : دو تا Iterable رو کنار هم قرار میده
    # dict() : حالا میاد و نتیجه zip که generator است رو تبدیل به دیکشنری میکنه که DIGITS_SOURCE کلید و DIGITS_TARGET مقدار است
    # ** : این همان Dictionary Unpacking است یعنی میاد و محتوای دیکشنری رو به صورت Literal می نویسه
    # حالا تو کد بالا محتوای دیکشنری ها کنار هم قرار میگیرن و یک دیکشنری کامل میسازند
    # و حالا maketrans() میاد و یک Translation Table قابل استفاده برای translate() میسازه
})
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
        if not isinstance(text, str): # اگر متن از جنس رشته نباشد
            raise TypeError(f"Expected str, got {type(text).__name__}") # آنگاه یک TypeError با متن مقابل به کاربر نشون بده
            # type(text) یک شی از نوع تایپ و یک property __name__ با مقدار اسم نوع text می باشد

        if not text.strip(): # اگر متن خالی بود یا None بود
            return "" # آنگاه خود متن که یا خالی است و یا None است را برمی گرداند
        
        return text.translate(_TRANSLATION_TABLE) # تبدیل کاراکتر و اعداد عربی به فارسی