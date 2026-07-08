"""مجموعه ای از نماد ها و علائم نگارشی"""

import unicodedata # این ماژول اطلاعات Unicode Character Database (UCD) را در اختیار ما می گذارد
import sys # این ماژول هم اطلاعات مربوط به خود مفسر پایتون را می‌ دهد

TEXT_SYMBOLS = frozenset( # همان set ولی Immutable یعنی دیگر نمی‌ توان چیزی به آن اضافه کرد
    ch # اگر شرط برقرار باشد این کاراکتر را داخل frozenset قرار می دهیم
    for codepoint in range(sys.maxunicode + 1) # تا آخرین code point unicode با کمک +1 را بررسی می کند
    if unicodedata.category(ch := chr(codepoint))[0] in {"P", "S"}
    # chr() : عدد را تبدیل به کاراکتر می‌کند
    # Walrus Operator : که میاد و مقدار ch رو برابر chr(codepoint) و همزمان مقدار رو بر می گردونه
    # unicodedata.category(ch) : برای هر کاراکتر Category را می‌ گیرد
    # [0] in {"P", "S"} : یعنی اولین حرف دسته باید P یا S باشه که یعنی یا علائم نگارشی باشه یا نماد
)