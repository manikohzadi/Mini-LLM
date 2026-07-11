------

# Roadmap (ثابت)
این Roadmap از ابتدا تا انتهای پروژه تغییر نخواهد کرد.
```
Phase 1
────────────
Foundation

constants.py
exceptions.py
types.py
metadata.py

↓

Phase 2
────────────
Core Vocabulary

vocabulary.py

↓

Phase 3
────────────
Builder

builders/

↓

Phase 4
────────────
Serializer

serializers/

↓

Phase 5
────────────
Statistics

statistics/

↓

Phase 6
────────────
Tests

↓

Phase 7
────────────
Documentation
```



# معماری

هر لایه فقط مسئول یک چیز است.

                    User
                      │
          ┌───────────┼────────────┐
          │           │            │
          ▼           ▼            ▼
     Vocabulary   Builder    Serializer
          │           │            │
          └───────────┼────────────┘
                      │
                Internal Utilities



# ساختار پوشه‌ها (نهایی)
```
vocab/
│
├── __init__.py
│
├── constants.py
├── exceptions.py
├── metadata.py
├── types.py
│
├── vocabulary.py
│
├── builders/
│   ├── __init__.py
│   ├── base.py
│   └── word.py
│
├── serializers/
│   ├── __init__.py
│   ├── base.py
│   └── json.py
│
├── statistics/
│   ├── __init__.py
│   └── vocabulary_statistics.py
│
├── _internal/
│   ├── __init__.py
│   ├── validation.py
│   └── helpers.py
│
└── py.typed
```