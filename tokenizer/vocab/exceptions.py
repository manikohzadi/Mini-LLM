"""
استثنا های سفارشی مورد استفاده توسط زیرسیستم واژگان.

هدف این ماژول ارائه یک سلسله مراتب ساختارمند برای تمام خطا های مربوط به واژگان است.

نکات
-----
- همه استثنا های سفارشی از ``VocabularyError`` ارث بری می کنند.
- استثنا های داخلی مانند ``TypeError``, ``ValueError`` و
  ``FileNotFoundError`` همچنان در صورت نیاز استفاده می شوند.
"""

from __future__ import annotations


class VocabularyError(Exception):
    # این کلاس معمولا مستقیما ایجاد نمی شود. هدف آن فراهم کردن یک ریشه
    # مشترک برای تمام خطا های اختصاصی زیرسیستم واژگان است.
    """
    کلاس پایه برای همه استثنا های مربوط به واژگان.
    """


# اعتبارسنجی


class ValidationError(VocabularyError):
    """
    زمانی رخ می دهد که اعتبارسنجی یک شی با شکست مواجه شود.
    """


class InvalidTokenError(ValidationError):
    """
    برای توکن هایی استفاده می شود که از نظر ساختاری نامعتبرند. مانند:

        - 123
        - None
        - ""
    """


class InvalidTokenIDError(ValidationError):
    """
    برای شناسه هایی استفاده می شود که از نظر ساختاری نامعتبرند. مانند:

        - -1
        - "2"
        - None
        - True
    """


class InvalidFrequencyError(ValidationError):
    """
    زمانی رخ می دهد که frequency با قرارداد نوع توکن سازگار نباشد.

    frequency توکن ویژه باید صفر باشد و frequency توکن عادی باید یک عدد
    صحیح مثبت باشد.
    """


class InvalidVocabularyError(ValidationError):
    """
    زمانی رخ می دهد که اجزای Vocabulary با یکدیگر سازگار نباشند.
    """


class InvalidMetadataError(ValidationError):
    """
    زمانی رخ می دهد که Metadata نامعتبر باشد. مانند زبان خالی, نام نامعتبر
    یا تاریخ نامعتبر.
    """


# جستجو


class VocabularyLookupError(VocabularyError):
    """
    کلاس پایه برای خطا های جستجو در Vocabulary است.

    استفاده از این نام مانع ایجاد تداخل مفهومی با استثنای داخلی
    ``LookupError`` پایتون می شود.
    """


class LookupError(VocabularyLookupError):
    """
    کلاس پایه سازگار با نسخه های قبلی برای خطا های مربوط به جستجو است.
    """


class UnknownTokenError(LookupError):
    """
    زمانی رخ می دهد که مقدار توکن معتبر است, ولی در Vocabulary وجود ندارد.
    """

    __slots__ = ("token",)

    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(f"Unknown token: {token!r}")


class UnknownTokenIDError(LookupError):
    """
    زمانی رخ می دهد که شناسه توکن در Vocabulary وجود نداشته باشد.
    """

    __slots__ = ("token_id",)

    def __init__(self, token_id: int) -> None:
        self.token_id = token_id
        super().__init__(f"Unknown token id: {token_id}")


# ============================================================================
# سازنده
# ============================================================================


class BuilderError(VocabularyError):
    """
    کلاس پایه برای خطا های مربوط به ساخت Vocabulary.
    """


class EmptyCorpusError(BuilderError):
    """
    زمانی رخ می دهد که ساخت Vocabulary از یک پیکره خالی درخواست شود.
    """


class DuplicateTokenError(BuilderError):
    """
    زمانی رخ می دهد که میان توکن های رزرو شده مقدار تکراری وجود داشته باشد.
    """


# ============================================================================
# مخزن
# ============================================================================


class RepositoryError(VocabularyError):
    """
    کلاس پایه برای خطا های مربوط به مخزن Vocabulary.
    """


class SaveVocabularyError(RepositoryError):
    """
    زمانی رخ می دهد که ذخیره کردن Vocabulary با شکست مواجه شود.
    """


class LoadVocabularyError(RepositoryError):
    """
    زمانی رخ می دهد که بارگذاری Vocabulary با شکست مواجه شود.
    """


# ============================================================================
# سریال ساز
# ============================================================================


class SerializerError(VocabularyError):
    """
    کلاس پایه برای خطا های مربوط به سریال سازی و بازیابی Vocabulary.
    """


class SerializationError(SerializerError, SaveVocabularyError):
    """
    زمانی رخ می دهد که تبدیل Vocabulary به داده قابل ذخیره با شکست مواجه شود.
    """


class DeserializationError(SerializerError, LoadVocabularyError):
    """
    زمانی رخ می دهد که بازسازی Vocabulary از داده ذخیره شده با شکست مواجه شود.
    """


class UnsupportedFormatError(SerializerError):
    """
    زمانی رخ می دهد که فرمت درخواست شده توسط سریال ساز پشتیبانی نشود.
    """


class VocabularyFileNotFoundError(
    DeserializationError,
    FileNotFoundError,
):
    """
    زمانی رخ می دهد که فایل Vocabulary در مسیر مورد نظر وجود نداشته باشد.
    """

    __slots__ = ("path",)

    def __init__(self, path: object) -> None:
        self.path = path
        super().__init__(f"Vocabulary file not found: {path}")


class InvalidVocabularyFormatError(DeserializationError):
    """
    زمانی رخ می دهد که داده ذخیره شده Vocabulary ساختار معتبر نداشته باشد.
    """


class UnsupportedVocabularyFormatVersionError(SerializerError):
    """
    زمانی رخ می دهد که داده سریال شده از نسخه فرمت پشتیبانی نشده استفاده کند.
    """

    __slots__ = ("found_version", "supported_version")

    def __init__(
        self,
        found_version: object,
        supported_version: int,
    ) -> None:
        self.found_version = found_version
        self.supported_version = supported_version
        super().__init__(
            "Unsupported vocabulary format version: "
            f"found {found_version!r}, supported {supported_version}."
        )


# ============================================================================
# وضعیت
# ============================================================================


class FrozenVocabularyError(VocabularyError):
    """
    زمانی رخ می دهد که تغییر دادن یک Vocabulary منجمد شده درخواست شود.
    """


__all__ = (
    "VocabularyError",
    "ValidationError",
    "InvalidTokenError",
    "InvalidTokenIDError",
    "InvalidFrequencyError",
    "InvalidVocabularyError",
    "InvalidMetadataError",
    "VocabularyLookupError",
    "LookupError",
    "UnknownTokenError",
    "UnknownTokenIDError",
    "BuilderError",
    "EmptyCorpusError",
    "DuplicateTokenError",
    "RepositoryError",
    "SaveVocabularyError",
    "LoadVocabularyError",
    "SerializerError",
    "SerializationError",
    "DeserializationError",
    "UnsupportedFormatError",
    "VocabularyFileNotFoundError",
    "InvalidVocabularyFormatError",
    "UnsupportedVocabularyFormatVersionError",
    "FrozenVocabularyError",
)
