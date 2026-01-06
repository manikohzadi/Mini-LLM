import re  # اضافه کردن کتابخانه regular expression برای بخش استفاده از الگو ها در text-cleaning
from typing import List  # اضافه کردن کتابخانه typing برای اضافه کردن type annotations به توابع و بالا بردن خوانایی کد
from tokenizer.punctuation_list import punctuation # اضافه کردن لیستی از تمام علائم نگارشی دنیا

punctuation_set = set(punctuation) # تبدیل لیست علائم نگارشی به مجموعه چرا که وقتی می خواهیم یک آیتم را ببینیم که درون علائم نگارشی هست از hash table استفاده کنه و سریعتر بشه

# الگوها
URL_PATTERN = re.compile(
    r"""
    (?P<scheme>     # Named capturing group با اسم scheme
        [a-zA-Z]                     # شروع با حرف
        [a-zA-Z0-9+.-]*              # ادامه‌ی مجاز scheme
    )
    ://                              # جداکننده‌ی scheme

    (?:     # Non capturing group : در خروجی ذخیره نمی شود
        (?P<userinfo>       # Named capturing group با اسم userinfo
            [^\s:@/?#]+       # username که نباید شامل whitespace و نماد هایی مثل :@/?# باشه
            (?::[^\s@/?#]+)?      # :password اختیاری
        )@
    )?      # userinfo اختیاری

    (?:     # Non capturing group
        (?P<ipv4>       # Named capturing group با اسم ipv4
            (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d) # توضیحات برای 4 بخش جدا شده با عملگر OR:
            #       بخش اول : عدد 25 باشه و بعدش یک عدد بین 0 تا 5 باشه
            #       بخش دوم : اول عدد 2 باشه بعد هم یک عدد بین 0 تا 4 بعد این هم یک عدد دیگر باشه
            #       بخش سوم : اول عدد 1 باشه بعد هم یک عدد دو رقمی
            #       بخش چهارم : اول یک عدد بین 1 تا 9 باشه یا می تونه نباشه بعد هم یک عدد دیگر باشه
            (?:\. # خود کاراکتر نقطه
            (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d) # توضیحات مثل بالا
            ){3} # سه بار گروه بالا تکرار شود
        )
        |
        (?P<ipv6>       # Named capturing group با اسم ipv6
            \[ # خود کاراکتر براکت
            (?:
            (?:[0-9a-fA-F]{1,4}:){7} # اعداد هگزادسیمال با طول بین 1 تا4 و یک علامت دونقطه بعدش که باید این هفت بار تکرار شده باشه
            [0-9a-fA-F]{1,4}| # بعد هم یک عدد هگزادسیمال دیگه با طول بین 1 تا 4 برای اتمام کار
             (?:[0-9a-fA-F]{1,4}:){1,7}:| # اعداد هگزادسیمال با طول بین 1 تا 4 و یک دونقطه هم بعدش و این هم باید بین 1 تا 7 بار تکرار بشه و یک دونقطه هم بعدش باشه
             ::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}) # توضیحات شبیه بالا
        )
        | # عملگر منطقی OR
        (?P<domain>     # Named capturing group با اسم domain
            (?:
                [a-zA-Z0-9]                       # شروع دامنه که می تونه شامل حروف بزرگ و کوچک انگلیسی و اعداد انگلیسی باشد
                [a-zA-Z0-9\-]{0,61}               # وسط دامنه که مثل بالا است و می تواند شامل علامت dash هم باشد و باید طولش بین 0 تا 61 باشد
                [a-zA-Z0-9]                       # انتهای دامنه توضیحات مثل قبل
                \.
            )+
            [a-zA-Z]{2,63}                        # TLD که فقط باید حرف باشه و طولش بین 2 تا 63 باشه مثل .com و .ir و .org
        )
    )

    (?::(?P<port>[0-9]{1,5}))?       # پورت اختیاری

    (?P<path>
        /[^\s?#]*                    # path
    )?

    (?P<query>
        \?[^\s#]*                    # query
    )?

    (?P<fragment>
        \#[^\s]*                     # fragment
    )?
    """,
    re.VERBOSE | re.IGNORECASE # IGNORECASE برای نادیده گرفتن بزرگی و کوچکی کلمات و VERBOSE برای نوشتن رجکس چند خطی و کامنت گذاشتن داخل آن برای خوانایی
) # الگوی مخصوص پیدا کردن URL که به بزرگی و کوچکی حروف اهمیتی نمی دهد و از قوانین RFC پیروی می کند.
EMAIL_PATTERN = re.compile(
    r"""
    (?P<local>                           # بخش local-part
        (?:                             
            [a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+   # حروف و نمادهای مجاز
            (?:\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*  # نقاط داخلی
        |
            "(?:\\[\x00-\x7F]|[^"\\])*"        # quoted-string
        )
    )
    @                                    # جداکننده
    (?P<domain>                          # بخش دامنه
        (?:                             
            [a-zA-Z0-9]                   # شروع هر label
            (?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])? # وسط و انتهای label
            (?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)* # ساب‌دامین‌ها
        )
        \.[a-zA-Z]{2,63}                  # TLD
    )
    """,
    re.VERBOSE | re.IGNORECASE
)

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
def separate_punctuations(text):
    text_list = []

    for character in text:
        if character in punctuation_set:
            text_list.append(f" {character} ")
        else:
            text_list.append(character)

    return ''.join(text_list)

# کاهش حروف تکراری
def normalize_repeated_chars(text: str) -> str:
    """حروف تکراری پشت سر هم را به یک تا کاهش می‌دهد"""
    return re.sub(r"(.)\1{2,}", r"\1", text)

def normalize_persian_unicode(text: str) -> str:
    # حذف کشیده (Tatweel)
    text = text.replace("\u0640", "")
    # حذف نیم‌فاصله (ZWNJ)
    text = text.replace("\u200c", "")
    return text

def clean_text(text: str) -> str:
    """متن را پاکسازی می‌کند"""
    text = text.lower()
    text = normalize_persian_unicode(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = normalize_repeated_chars(text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """توکنایزر word-level برای فارسی مبتنی بر فاصله و علائم نگارشی"""
    text = clean_text(text)
    # جدا کردن علائم نگارشی از کلمات
    text = separate_punctuations(text)
    tokens = text.split()
    return tokens