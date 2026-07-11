"""
Core vocabulary implementation.

The Vocabulary class is the central domain object of the vocabulary
subsystem. It provides fast token ↔ id lookup, frequency tracking,
controlled mutation, and metadata management.

Notes
-----
- Building a vocabulary is the responsibility of builders.
- Serialization is the responsibility of serializers.
- Persistence is the responsibility of repositories.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from types import MappingProxyType

from .constants import (
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_ID,
    UNK_TOKEN,
)
from .exceptions import (
    FrozenVocabularyError,
    InvalidTokenError,
    InvalidTokenIDError,
    UnknownTokenError,
    UnknownTokenIDError,
)
from .metadata import VocabularyMetadata
from .types import (
    Frequency,
    IDToToken,
    Token,
    TokenFrequencies,
    TokenID,
    TokenSequence,
    TokenToID,
)


@dataclass(slots=True)
class Vocabulary:
    """
    Bidirectional vocabulary.

    This class owns the mapping between tokens and token IDs.

    Complexity
    ----------
    token -> id : O(1)
    id -> token : O(1)
    """

    _token_to_id: TokenToID

    _id_to_token: IDToToken

    _frequencies: TokenFrequencies

    metadata: VocabularyMetadata = field(
        default_factory=VocabularyMetadata
    )

    _frozen: bool = False

    @classmethod
    def empty(
        cls,
        metadata: VocabularyMetadata | None = None,
    ) -> "Vocabulary":
        """
        Create an empty vocabulary containing only special tokens.
        """

        token_to_id: TokenToID = {}

        id_to_token: IDToToken = []

        frequencies: TokenFrequencies = {}

        for token in SPECIAL_TOKENS:
            token_id = TokenID(len(id_to_token))

            token_to_id[token] = token_id
            id_to_token.append(token)
            frequencies[token] = 0

        return cls(
            _token_to_id=token_to_id,
            _id_to_token=id_to_token,
            _frequencies=frequencies,
            metadata=metadata or VocabularyMetadata(),
        )
    
    @property
    def token_to_id_map(self) -> TokenToID:
        """Return a read-only view of the token→id mapping."""
        return MappingProxyType(self._token_to_id)

    @property
    def id_to_token_map(self) -> tuple[Token, ...]:
        """
        Return a read-only view of the ID → token mapping.
        """

        return tuple(self._id_to_token)

    @property
    def frequencies(self) -> TokenFrequencies:
        """Return a read-only view of token frequencies."""
        return MappingProxyType(self._frequencies)
    
    @property
    def frozen(self) -> bool:
        """
        Whether the vocabulary is frozen.
        """
        return self._frozen

    @property
    def size(self) -> int:
        """
        Total vocabulary size.
        """
        return len(self._id_to_token)

    def __len__(self) -> int:
        return self.size
    
    def __iter__(self) -> Iterator[Token]:
        """
        Iterate over vocabulary tokens.
        """
        return iter(self._id_to_token)

    def __contains__(self, token: Token) -> bool:
        """
        Return True if token exists.
        """
        return token in self._token_to_id
    
    # ========================================================================
    # Lookup
    # ========================================================================

    def token_to_id(self, token: Token) -> TokenID:
        """
        Return the ID corresponding to a token.

        Raises
        ------
        UnknownTokenError
            If the token does not exist.
        """

        try:
            return self._token_to_id[token]
        except KeyError as exc:
            raise UnknownTokenError(token) from exc

    def id_to_token(self, token_id: TokenID) -> Token:
        """
        Return the token corresponding to an ID.

        Raises
        ------
        UnknownTokenIDError
            If the ID does not exist.
        """

        try:
            return self._id_to_token[token_id]
        except IndexError as exc:
            raise UnknownTokenIDError(token_id) from exc
        
    # ========================================================================
    # Encoding
    # ========================================================================

    def encode(
        self,
        tokens: TokenSequence,
        *,
        unknown_token_id: TokenID = UNK_ID,
    ) -> list[TokenID]:
        """
        Encode a sequence of tokens into token IDs.

        Unknown tokens are mapped to ``unknown_token_id``.
        """

        return [
            self._token_to_id.get(token, unknown_token_id)
            for token in tokens
        ]

    def decode(
        self,
        token_ids: Iterable[TokenID],
    ) -> list[Token]:
        """
        Decode token IDs into tokens.

        Raises
        ------
        UnknownTokenIDError
            If an ID is invalid.
        """

        return [
            self.id_to_token(token_id)
            for token_id in token_ids
        ]
    
    # ========================================================================
    # Frequency
    # ========================================================================

    def frequency(self, token: Token) -> Frequency:
        """
        Return the frequency of a token.

        Raises
        ------
        UnknownTokenError
            If the token does not exist.
        """

        try:
            return self._frequencies[token]
        except KeyError as exc:
            raise UnknownTokenError(token) from exc
        
    def contains_id(self, token_id: TokenID) -> bool:
        """
        Return True if the given ID exists.
        """

        return 0 <= token_id < len(self._id_to_token)
    
    @property
    def pad_token(self) -> Token:
        return PAD_TOKEN

    @property
    def unk_token(self) -> Token:
        return UNK_TOKEN

    @property
    def bos_token(self) -> Token:
        return BOS_TOKEN

    @property
    def eos_token(self) -> Token:
        return EOS_TOKEN

    @property
    def pad_id(self) -> TokenID:
        return PAD_ID

    @property
    def unk_id(self) -> TokenID:
        return UNK_ID

    @property
    def bos_id(self) -> TokenID:
        return BOS_ID

    @property
    def eos_id(self) -> TokenID:
        return EOS_ID
    
    # ========================================================================
    # Mutation
    # ========================================================================

    def add_token(
        self,
        token: Token,
        *,
        frequency: Frequency = 0,
    ) -> TokenID:
        """
        Add a token to the vocabulary.

        If the token already exists, its existing ID is returned.

        Raises
        ------
        FrozenVocabularyError
            If the vocabulary is frozen.

        InvalidTokenError
            If the token is empty.
        """

        if self._frozen:
            raise FrozenVocabularyError(
                "Vocabulary is frozen."
            )

        if not token:
            raise InvalidTokenError(
                "Token cannot be empty."
            )

        if token in self._token_to_id:
            return self._token_to_id[token]

        token_id = TokenID(len(self._id_to_token))

        self._token_to_id[token] = token_id

        self._id_to_token.append(token)

        self._frequencies[token] = frequency

        return token_id
    
    def extend(
        self,
        tokens: Iterable[Token],
    ) -> None:
        """
        Add multiple tokens.

        Existing tokens are ignored.
        """

        for token in tokens:
            self.add_token(token)

    # ========================================================================
    # Freeze
    # ========================================================================

    def freeze(self) -> None:
        """
        Freeze the vocabulary.

        After freezing, no token can be added.
        """

        self._frozen = True

    @property
    def is_empty(self) -> bool:
        """
        Return whether the vocabulary contains
        only the special tokens.
        """

        return len(self) == len(SPECIAL_TOKENS)
    
    def copy(self) -> "Vocabulary":
        """
        Return a deep copy of the vocabulary.
        """

        return Vocabulary(
            _token_to_id=self._token_to_id.copy(),
            _id_to_token=self._id_to_token.copy(),
            _frequencies=self._frequencies.copy(),
            metadata=self.metadata,
            _frozen=self._frozen,
        )
    
    # ========================================================================
    # Export
    # ========================================================================

    def tokens(self) -> tuple[Token, ...]:
        """
        Return all tokens ordered by their IDs.
        """

        return tuple(self._id_to_token)

    def token_ids(self) -> range:
        """
        Return an iterable over all token IDs.
        """

        return range(len(self._id_to_token))

    def items(self) -> Iterator[tuple[Token, TokenID]]:
        """
        Iterate over (token, token_id) pairs.
        """

        return iter(self._token_to_id.items())
    
    # ========================================================================
    # Statistics
    # ========================================================================

    @property
    def vocabulary_size(self) -> int:
        """
        Return the total vocabulary size.
        """

        return len(self._id_to_token)

    @property
    def special_token_count(self) -> int:
        """
        Return the number of reserved special tokens.
        """

        return len(SPECIAL_TOKENS)

    @property
    def normal_token_count(self) -> int:
        """
        Return the number of non-special tokens.
        """

        return self.vocabulary_size - self.special_token_count
    
    # ========================================================================
    # Validation
    # ========================================================================

    def has_token(
        self,
        token: Token,
    ) -> bool:
        """
        Return whether the token exists.
        """

        return token in self._token_to_id

    def has_token_id(
        self,
        token_id: TokenID,
    ) -> bool:
        """
        Return whether the token ID exists.
        """

        return 0 <= token_id < len(self._id_to_token)
    
    # ========================================================================
    # Magic Methods
    # ========================================================================

    def __repr__(self) -> str:
        """
        Return the official string representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"size={self.vocabulary_size}, "
            f"frozen={self._frozen})"
        )

    def __bool__(self) -> bool:
        """
        Return True if the vocabulary contains
        at least one non-special token.
        """

        return not self.is_empty
    

    # ========================================================================
    # Serialization Interface
    # ========================================================================

    def to_dict(self) -> dict[str, object]:
        """
        Convert the vocabulary into a serializable dictionary.
        """

        return {
            "metadata": self.metadata.to_dict(),
            "token_to_id": dict(self._token_to_id),
            "id_to_token": list(self._id_to_token),
            "frequencies": dict(self._frequencies),
            "frozen": self._frozen,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "Vocabulary":
        """
        Construct a vocabulary from a dictionary.
        """

        metadata = VocabularyMetadata.from_dict(data["metadata"])

        token_to_id = {
            Token(token): TokenID(token_id)
            for token, token_id in data["token_to_id"].items()
        }

        id_to_token = [
            Token(token)
            for token in data["id_to_token"]
        ]

        frequencies = {
            Token(token): Frequency(freq)
            for token, freq in data["frequencies"].items()
        }

        return cls(
            _token_to_id=token_to_id,
            _id_to_token=id_to_token,
            _frequencies=frequencies,
            metadata=metadata,
            _frozen=bool(data["frozen"]),
        )
    
    # ========================================================================
    # Equality
    # ========================================================================

    def __eq__(
        self,
        other: object,
    ) -> bool:
        """
        Compare two vocabularies.
        """

        if not isinstance(other, Vocabulary):
            return NotImplemented

        return (
            self._token_to_id == other._token_to_id
            and self._id_to_token == other._id_to_token
            and self._frequencies == other._frequencies
            and self.metadata == other.metadata
            and self._frozen == other._frozen
        )
    
    # ========================================================================
    # Utility
    # ========================================================================

    def clear(self) -> None:
        """
        Remove every non-special token from the vocabulary.

        Raises
        ------
        FrozenVocabularyError
            If the vocabulary is frozen.
        """

        if self._frozen:
            raise FrozenVocabularyError(
                "Vocabulary is frozen."
            )

        self._token_to_id.clear()
        self._id_to_token.clear()
        self._frequencies.clear()

        for token in SPECIAL_TOKENS:
            token_id = TokenID(len(self._id_to_token))

            self._token_to_id[token] = token_id
            self._id_to_token.append(token)
            self._frequencies[token] = 0

    def __reversed__(self) -> Iterator[Token]:
        """
        Iterate over the vocabulary in reverse ID order.
        """

        return reversed(self._id_to_token)

    def __sizeof__(self) -> int:
        """
        Return an approximate memory usage.
        """

        size = object.__sizeof__(self)

        size += self._token_to_id.__sizeof__()
        size += self._id_to_token.__sizeof__()
        size += self._frequencies.__sizeof__()

        return size
    
    # ========================================================================
    # End of class
    # ========================================================================