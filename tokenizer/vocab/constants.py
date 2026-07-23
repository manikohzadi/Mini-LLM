"""
این فایل تمام مقادیر ثابت مشترک Vocabulary را تعریف می کند
"""

from __future__ import annotations # این Import رفتار Type Annotation ها را مدرن تر می کند.
# در نسخه های قدیمی تر Python, بعضی Annotation ها بلافاصله هنگام تعریف کلاس یا تابع ارزیابی می شدند. با این دستور، Annotation ها به شکل تأخیری مدیریت می شوند.
# مثلا می توان راحت تر به کلاسی اشاره کرد که هنوز کامل تعریف نشده است
#   def copy(self) -> "Vocabulary":
#       ...
# در constants.py نیاز مستقیمی به این قابلیت دیده نمی شود, اما استفاده از آن به صورت یک استاندارد یکسان در کل پکیج مورد نظر ما است.
# بعد از این Import وقتی برای تابع یا کلاسی annotation ی تعریف کنیم وقتی property __annotations__ را نگاه کنیم type ها به صورت Stringized Annotations در آمدند.

from typing import Final # Final به Type Checker می گوید متغیر نباید دوباره مقداردهی شود.
# Final بیشتر یک قرارداد ایستا است, نه محافظ Runtime.
# یعنی Python از نظر Runtime همچنان اجازه می دهد که: PAD_ID = 10
# ولی mypy یا ابزار مشابه آن را خطا گزارش می کند.

# Library

VOCAB_FILE_ENCODING: Final[str] = "utf-8" # انکودینگ ثابت فایل Vocabulary در Serializer استاندارد پروژه است

DEFAULT_ENCODING: Final[str] = VOCAB_FILE_ENCODING # انکودینگ پیش فرض که برای ذخیره سازی توکن های زبان فارسی مهم است

VOCAB_FORMAT_VERSION: Final[int] = 1 # نسخه فرمت که اگر در آینده ساختار فایل تغییر کنه می تونیم افزایشش بدیم و این کار برای سازگاری نسخه ها مفید است.

# Special Tokens

PAD_TOKEN: Final[str] = "<PAD>" # مخفف Padding است و برای هم اندازه کردن توالی ها استفاده می شود.

UNK_TOKEN: Final[str] = "<UNK>" # مخفف Unknown است و وقتی توکنی داخل Vocabulary وجود ندارد, می‌توان آن را به <UNK> تبدیل کرد.

BOS_TOKEN: Final[str] = "<BOS>" # مخفف Beginning of Sequence است و نشانه شروع توالی است و مدل می‌تواند با دیدن آن بفهمد تولید جمله از کجا آغاز شده است.

EOS_TOKEN: Final[str] = "<EOS>" # مخفف End of Sequence است و نشانه پایان توالی است و در مدل مولد, تولید <EOS> معمولا به این معناست که مدل باید تولید متن را متوقف کند.

SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
) # یک تاپل از توکن های ویژه که انتخاب بهتری از لیست می باشد چون نشان می دهد تغییرناپذیری و ترتیب مهم است

SPECIAL_TOKEN_COUNT: Final[int] = len(SPECIAL_TOKENS) # تعداد توکن های ویژه را محاسبه می کند و مزیتش این است اگر توکن ویژه جدیدی مثل <MASK> اضافه شود به جای نوشتن مستقیم عدد 4 طول تاپل توکن های ویژه رو محاسبه می کنیم.

# Reserved IDs
# این چهار مقدار قرارداد عددی Vocabulary هستند و ثابت بودن این شناسه ها اهمیت زیادی دارد.

PAD_ID: Final[int] = 0

UNK_ID: Final[int] = 1

BOS_ID: Final[int] = 2

EOS_ID: Final[int] = 3

SPECIAL_TOKEN_ID_PAIRS: Final[tuple[tuple[str, int], ...]] = (
    (PAD_TOKEN, PAD_ID),
    (UNK_TOKEN, UNK_ID),
    (BOS_TOKEN, BOS_ID),
    (EOS_TOKEN, EOS_ID),
)

SPECIAL_TOKEN_IDS: Final[tuple[int, ...]] = tuple(
    token_id for _, token_id in SPECIAL_TOKEN_ID_PAIRS
)

# Builder Defaults

DEFAULT_MIN_FREQUENCY: Final[int] = 1 # حداقل تعداد تکرار لازم برای ورود توکن به Vocabulary است.

DEFAULT_MAX_VOCAB_SIZE: Final[int] = 100_000 # حداکثر تعداد توکن های عادی Vocabulary را به جز توکن های ویژه مشخص می کند

DEFAULT_MAX_NORMAL_TOKEN_COUNT: Final[int] = DEFAULT_MAX_VOCAB_SIZE
# نکته: _ فقط برای خوانایی کد است و پایتون همان 100000 را در نظر می گیرد

# Serialization
# ثابت های مربوط به ذخیره سازی

JSON_INDENT: Final[int] = 4 # مقدار تورفتگی JSON

JSON_EXTENSION: Final[str] = ".json" # پسوند استاندارد فایل JSON را تعریف می کند.

# Public API
# این قسمت مشخص می کند هنگام Import ستاره ای, چه نام هایی عمومی محسوب شوند.

__all__ = (
    "VOCAB_FILE_ENCODING",
    "DEFAULT_ENCODING",
    "VOCAB_FORMAT_VERSION",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKEN_COUNT",
    "PAD_ID",
    "UNK_ID",
    "BOS_ID",
    "EOS_ID",
    "SPECIAL_TOKEN_ID_PAIRS",
    "SPECIAL_TOKEN_IDS",
    "DEFAULT_MIN_FREQUENCY",
    "DEFAULT_MAX_VOCAB_SIZE",
    "DEFAULT_MAX_NORMAL_TOKEN_COUNT",
    "JSON_INDENT",
    "JSON_EXTENSION",
)
# حتی اگر نامی داخل __all__ نباشد, همچنان ممکن است نوشته شود
# پس __all__ بیشتر مشخص کننده API رسمی و رفتار import * است, نه یک سیستم امنیتی یا دسترسی خصوصی.

# دو تفکیک مهم:
# Annotation اطلاعاتی است که داخل کد ثبت می شود.
# Type Hint یکی از کاربرد های Annotation است که می گوید: نوع مورد انتظار متغیر, ویژگی کلاس, پارامتر یا خروجی چیست
# مثلا Annotation الزاما مجبور نیست برای Type Checking باشد می تواند مقداری مثل "processed value" باشد.