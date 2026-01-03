import re  # اضافه کردن کتابخانه regular expression برای بخش استفاده از الگو ها در text-cleaning
from typing import List, Optional  # اضافه کردن کتابخانه typing برای اضافه کردن type annotations به توابع و بالا بردن خوانایی کد

# الگوها
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002B00-\U00002BFF"
    "]+",
    flags=re.UNICODE,
)

# علائم نگارشی (فارسی و انگلیسی) - توکن جدا
PUNCT_PATTERN = re.compile(r"([،؛؟!?.،؛«»…—\-]+)")

# کاهش حروف تکراری
def normalize_repeated_chars(text: str) -> str:
    """حروف تکراری پشت سر هم را به یک تا کاهش می‌دهد"""
    return re.sub(r"(.)\1{2,}", r"\1", text, flags=re.UNICODE)


def clean_text(text: Optional[str]) -> str:
    """متن را پاکسازی می‌کند"""
    if text is None:
        return ""
    if not isinstance(text, str):
        # در صورتی که ورودی نوع متفاوتی داشت، آن را به رشته تبدیل می‌کنیم
        text = str(text)

    text = text.lower()
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)  # فاصله‌ها را استاندارد می‌کند
    text = normalize_repeated_chars(text)
    return text.strip()


def tokenize(text: Optional[str]) -> List[str]:
    """توکنایزر word-level برای فارسی مبتنی بر فاصله و علائم نگارشی"""
    text = clean_text(text)
    if not text:
        return []
    # جدا کردن علائم نگارشی از کلمات
    text = PUNCT_PATTERN.sub(r" \1 ", text)
    tokens = [t for t in text.split() if t]
    return tokens
