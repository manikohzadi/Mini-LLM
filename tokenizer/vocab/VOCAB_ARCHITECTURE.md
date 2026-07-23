# Vocabulary architecture

## Stable roadmap

```text
Phase 1
Foundation

constants.py
exceptions.py
types.py
metadata.py

Phase 2
Core vocabulary

vocabulary.py

Phase 3
Builders

builders/

Phase 4
Serializers

serializers/

Phase 5
Statistics

statistics/

Phase 6
Tests

Phase 7
Documentation
```

## Responsibility boundaries

```text
                    User
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
     Vocabulary    Builder    Serializer
          |           |           |
          +-----------+-----------+
                      |
               Internal utilities
```

- `Vocabulary` owns token and ID mappings, frequencies, metadata and frozen state.
- `WordVocabularyBuilder` validates and builds a frozen vocabulary from a tokenized corpus.
- `JSONVocabularySerializer` handles versioned JSON conversion and atomic file persistence.
- `VocabularyStatistics` calculates read-only statistics outside the core domain object.
- `_internal` contains private validation and state construction helpers.

## Compatibility layer

The new architecture remains the canonical implementation. Older public names are retained as working adapters so existing callers do not break.

- `DefaultVocabularyBuilder` delegates to `WordVocabularyBuilder`.
- `JSONVocabularyRepository` delegates to `JSONVocabularySerializer`.
- `VocabularyFactory` continues to work through the compatibility builder and repository interfaces.
- `validators.py` delegates to the centralized validator in `_internal/validation.py`.
- `LookupError` remains available below `VocabularyLookupError` so old exception handlers continue to catch lookup failures.
- Repository exceptions remain active through the serializer exception hierarchy.
- `DEFAULT_MAX_VOCAB_SIZE` remains available beside `DEFAULT_MAX_NORMAL_TOKEN_COUNT`.

## Frequency contract

The frequency field has one exact meaning: the number of corpus observations represented by a token entry.

- Reserved special tokens always have frequency `0` because they are created by the vocabulary system rather than selected from corpus counts.
- Every normal token must have a strictly positive frequency.
- `WordVocabularyBuilder` already satisfies this rule because selected corpus tokens have frequency at least `min_frequency`, and `min_frequency` is at least `1`.
- `Vocabulary.add_token` requires an explicit positive frequency when a new normal token is added.
- Calling `add_token` for an existing token without a frequency remains an idempotent lookup and returns the existing ID.
- Supplying a different frequency for an existing token is rejected instead of silently discarding conflicting data.
- `Vocabulary.extend` accepts a mapping or an iterable of `(token, frequency)` pairs so every inserted normal token has an explicit frequency.
- Deserialization rejects persisted vocabularies containing a normal token with frequency `0`.

## JSON contract

- `VOCAB_FILE_ENCODING` is the fixed UTF-8 encoding of the standard JSON serializer.
- `DEFAULT_ENCODING` remains only as a compatibility alias for older callers.
- `VOCAB_FORMAT_VERSION` identifies the one persisted schema version supported by this release.
- Builders do not own file encoding or format version decisions.
- `VocabularyMetadata` exposes the current format version as read-only data and callers cannot override it.
- A different encoding or schema requires a different serializer implementation or a formal format migration.
- Missing, malformed, older and newer unsupported versions are rejected.
- Paths without a suffix receive `JSON_EXTENSION`.
- Conflicting suffixes are rejected.
- Files are written through a temporary file and atomically replaced after a successful flush.
- Loaded data is validated before a `Vocabulary` instance is returned.

## Package layout

```text
vocab/
|
|-- __init__.py
|-- constants.py
|-- exceptions.py
|-- metadata.py
|-- types.py
|-- vocabulary.py
|-- factory.py
|-- validators.py
|
|-- builders/
|   |-- __init__.py
|   |-- base.py
|   |-- word.py
|   `-- default_builder.py
|
|-- serializers/
|   |-- __init__.py
|   |-- base.py
|   `-- json.py
|
|-- repositories/
|   |-- __init__.py
|   |-- base.py
|   `-- json_repository.py
|
|-- statistics/
|   |-- __init__.py
|   `-- vocabulary_statistics.py
|
|-- _internal/
|   |-- __init__.py
|   |-- validation.py
|   `-- helpers.py
|
`-- py.typed
```

## Deliberate extension points

`EmptyCorpusError` is used when `WordVocabularyBuilder(require_non_empty=True)` is selected. The default remains permissive because a special-token-only vocabulary is valid.

The shared type aliases in `types.py` remain available for future tokenizer and training integrations. They do not add runtime behavior and are intentionally lightweight.
