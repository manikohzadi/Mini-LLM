"""
Industrial vocabulary module.
"""

from .factory import VocabularyFactory

from .metadata import VocabularyMetadata

from .vocabulary import Vocabulary

from .builders.default_builder import (
    DefaultVocabularyBuilder,
)

from .repositories.json_repository import (
    JSONVocabularyRepository,
)

__all__ = (
    "Vocabulary",
    "VocabularyFactory",
    "VocabularyMetadata",
    "DefaultVocabularyBuilder",
    "JSONVocabularyRepository",
)