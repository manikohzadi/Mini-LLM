"""
Industrial vocabulary module.

تصویر کلی معماری vocab: وظیفه این بخش, نگهداری ارتباط میان توکن ها و شناسه های عددی آن ها است.

- Vocabulary خود واژگان است.
- Builder واژگان را از روی داده ها می سازد.
- Repository آن را در فایل ذخیره یا از فایل بارگذاری می‌ کند.
- Factory استفاده از Builder و Repository را ساده می‌ کند.
- Validator ورودی‌ ها را کنترل می‌کند.
- Metadata اطلاعات جانبی واژگان را نگه می‌ دارد.
- types.py نوع‌ های مشترک را تعریف می‌ کند.
- exceptions.py خطا های اختصاصی این بخش را تعریف می کند.
- constants.py مقادیر ثابت را نگه می دارد.
- py.typed به ابزار های بررسی نوع, مثل mypy و pyright, اعلام می کند که این پکیج دارای Type Annotation های قابل استفاده است.
"""

from .builders import (
    DefaultVocabularyBuilder,
    VocabularyBuilder,
    WordVocabularyBuilder,
)
from .factory import VocabularyFactory
from .metadata import VocabularyMetadata
from .repositories import (
    JSONVocabularyRepository,
    VocabularyRepository,
)
from .serializers import (
    JSONVocabularySerializer,
    VocabularySerializer,
)
from .statistics import VocabularyStatistics
from .vocabulary import Vocabulary

__all__ = (
    "Vocabulary",
    "VocabularyFactory",
    "VocabularyMetadata",
    "VocabularyBuilder",
    "DefaultVocabularyBuilder",
    "WordVocabularyBuilder",
    "VocabularyRepository",
    "JSONVocabularyRepository",
    "VocabularySerializer",
    "JSONVocabularySerializer",
    "VocabularyStatistics",
)
