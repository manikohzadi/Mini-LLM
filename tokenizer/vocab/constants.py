"""
این فایل تمام مقادیر ثابت مشترک Vocabulary را تعریف می کند
"""

from __future__ import annotations

from typing import Final # Final به Type Checker می گوید متغیر نباید دوباره مقداردهی شود.
# Final بیشتر یک قرارداد ایستا است, نه محافظ Runtime.
# یعنی Python از نظر Runtime همچنان اجازه می دهد که: PAD_ID = 10
# ولی mypy یا ابزار مشابه آن را خطا گزارش می کند.

# Library

DEFAULT_ENCODING: Final[str] = "utf-8" # انکودینگ پیش فرض که برای ذخیره سازی توکن های زبان فارسی مهم است

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
)

SPECIAL_TOKEN_COUNT: Final[int] = len(SPECIAL_TOKENS)

# Reserved IDs

PAD_ID: Final[int] = 0

UNK_ID: Final[int] = 1

BOS_ID: Final[int] = 2

EOS_ID: Final[int] = 3

# Builder Defaults

DEFAULT_MIN_FREQUENCY: Final[int] = 1

DEFAULT_MAX_VOCAB_SIZE: Final[int] = 100_000

# Serialization

JSON_INDENT: Final[int] = 4

JSON_EXTENSION: Final[str] = ".json"

# Public API

__all__ = (
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
    "DEFAULT_MIN_FREQUENCY",
    "DEFAULT_MAX_VOCAB_SIZE",
    "JSON_INDENT",
    "JSON_EXTENSION",
)