import re # اضافه کردن کتابخانه regular expression برای بخش استفاده از الگو ها در text-cleaning
from typing import List # اضافه کردن کتابخانه typing برای اضافه کردن type annotations به توابع و بالا بردن خوانایی کد

# الگوها
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE) # این الگو تمام لینک‌های شروع‌شده با http://, https:// یا www. را در متن پیدا می‌کند، بدون توجه به حروف بزرگ و کوچک.
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE) # این الگو هر آدرس ایمیلی را که شامل نام کاربری، علامت @ و دامنه معتبر باشد پیدا می‌کند، بدون توجه به حروف بزرگ و کوچک.
HTML_TAG_PATTERN = re.compile(r"<[^>]+>") # این الگو هر تگ HTML را که با < شروع و با > پایان می‌یابد پیدا می‌کند، بدون توجه به محتوای داخلی تگ.
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
) # این الگو تمام ایموجی‌ها و نمادهای یونیکد تصویری داخل متن را پیدا یا حذف می‌کند با استفاده از unicode-ranges

# علائم نگارشی (اضافه کردن علائم رایج فارسی)
PUNCT_PATTERN = re.compile(r"[^\w\s\u0600-\u06FF]") # هر چیزی که نه حرف، نه عدد، نه فاصله، و نه حروف فارسی باشد را پیدا می‌کند


# کاهش حروف تکراری
def normalize_repeated_chars(text: str) -> str:
    """حروف تکراری پشت سر هم را به دو تا کاهش می‌دهد"""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def clean_text(text: str) -> str:
    """متن را پاکسازی می‌کند"""
    text = text.lower()
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = normalize_repeated_chars(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """توکنایزر word-level برای فارسی مبتنی بر فاصله"""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    return tokens