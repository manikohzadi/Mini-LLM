import re  # اضافه کردن کتابخانه regular expression برای بخش استفاده از الگو ها در text-cleaning
from typing import List, Tuple, Dict  # اضافه کردن کتابخانه typing برای type annotations
from tokenizer.punctuation_list import punctuation  # اضافه کردن لیستی از تمام علائم نگارشی دنیا

punctuation_set = set(punctuation)  # تبدیل بیست علائم نگارشی به set برای سریعتر شدن جست و جو در آن به دلیل استفاده set از hash table

URL_PATTERN = re.compile( # این تابع یک الگو را به Pattern Object تبدیل می کند که بتوانیم از آن هزار بار و خیلی سریعتر استفاده کنیم
    r"""
        (?P<scheme> # Named capturing group با اسم scheme
            [a-zA-Z] # شروع با حروف بزرگ و کوچک انگلیسی
            [a-zA-Z0-9+.-]* # ادامه‌ی مجاز scheme که می تواند شامل حروف بزرگ و کوچک و اعداد انگلیسی و علامت های +.- باشد
        ):// # جدا کننده scheme که آن را با دامنه جدا می کند

    (?: # Non capturing group : در خروجی ذخیره نمی شود
        (?P<userinfo> # Named capturing group با اسم userinfo
            [^\s:@/?#]+ # username که نباید شامل whitespace و نماد هایی مثل :@/?# باشه
            (?::[^\s@/?#]+)? # :password اختیاری است و نباید شامل چیز هایی که در بالا گفتیم باشه
        )@ # جداکننده userinfo با دامنه
    )? # userinfo اختیاری

    (?: # Non capturing group
        (?P<ipv4> # Named capturing group با اسم ipv4
            (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d) # توضیحات هر بخش جدا شده با عملگر منطقی OR(|):
            #       بخش اول : اول عدد 25 آمده باشه بعدش هم یک عددی که بین 0 تا 5 هست آمده باشه.
            #       بخش دوم : اول عدد 2 آمده باشه بعد هم عددی بین 0 تا 4 آمده یاشه بعد هم یک عدد بین 0 تا 9 آمده باشه.
            #       بخش سوم : اول عدد 1 آمده باشه بعد هم یک عدد دو رقمی آمده باشه
            #       بخش چهارم : اول یک عدد بین 1 تا 9 باشه یا می تونه نباشه بعد هم یک عدد بین 0 تا 9 باشه که یعنی اینکه ipv4 می تونه با عدد تک رقمی هم شروع بشه
            (?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3} # اول یک نقطه باشه بعدش هم توضیحات مثل بالا بعدش هم این 3 بار تکرار شه
        )
        | # یا
        (?P<ipv6> # Named capturing group با اسم ipv6
            \[ # خود کاراکتر براکت
            (?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4} # اول یک عدد هگزادسیمال با طول بین 1 تا 4 باشه بعدش هم یک علامت دو نقطه باشه و این 7 بار تکرار بشه بعدش یک عدد هگزادسیمال دیگه با طول بین 1 تا 4 برای پایان کار
            | # یا
             (?:[0-9a-fA-F]{1,4}:){1,7}: # اول یک عدد هگزادسیمال با طول بین 1 تا 4 بعدش هم یک دو نقطه باشه و این هم بین 1 تا 7 بار تکرار بشه و برای پایان کار هم یک دونقطه بیاد
             | # یا
             ::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}) # اول دو تا دونه دو نقطه باشه. بعدش هم یک عدد هگزادسیمال با طول بین 1 تا4 باشه بعدش هم یک دونه دو نقطه باشه و این روند بین 0 تا 6 بار تکرار بشه و پایان کار هم یک عدد هگزادسیمال با طول بین 1 تا 4 بیاد
            \] # خود کاراکتر براکت
        )
        | # یا
        (?P<domain> # Named capturing group با اسم domain
            (?:
                [a-zA-Z0-9] # شروع دامنه که می تونه با حروف بزرگ و کوچک و اعداد انگلیسی باشه
                [a-zA-Z0-9\-]{0,61} # وسط دامنه که مثل بالا است و می تواند شامل علامت dash باشد و تا طول بین 0 تا 61 کاراکتر ادامه داشته باشد.
                [a-zA-Z0-9] # انتهای دامنه که مثل همون شروع دامنه است
                \. # خود علامت نقطه
            )+ # این می تونه 1 یا بیشتر بار تکرار بشه
            [a-zA-Z]{2,63} # TLD مثل com, org, و ir که فقط می تونه شامل حروف بزرگ و کوچک انگلیسی باشه و تا طول بین 2 تا 63 کاراکتر ادامه داشته باشه.
        )
    )

    (?:: # علامت دو نقطه
        (?P<port> # Named capturing group با اسم port
        [0-9]{1,5}) # پورت : که می تونه یک عدد با طول بین 1 تا 5 باشه
    )? # اختیاری

    (?P<path> # Named capturing group با اسم path
        /[^\s?#]* # path : که اولش یک اسلش و بعدش هم فقط نباید شامل whitespace و ?# باشه
    )? # اختیاری

    (?P<query> # Named capturing group با اسم query
        \?[^\s#]* # query : که باید با علامت سوال سروع بشه و شامل whitespace و هش تگ نباشه
    )? # اختیاری

    (?P<fragment> # Named capturing group با اسم fragment
        \#[^\s]* # fragment : که باید با هش تگ شروع بشه و فقط شامل whitespace نباشه
    )? # اختیاری
    """,
    re.VERBOSE | re.IGNORECASE # IGNORECASE برای نادیده گرفتن بزرگی و کوچکی حروف و VERBOSE برای نوشتن رجکس چند خطی و کامنت گذاشتن داخل آن
)

EMAIL_PATTERN = re.compile(
    r"""
    (?P<local> # بخش local-part
        (?:                             
            [a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+
            (?:\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*
        |
            "(?:\\[\x00-\x7F]|[^"\\])*"
        )
    )
    @ # جداکننده
    (?P<domain> # بخش دامنه
        (?:                             
            [a-zA-Z0-9]
            (?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?
            (?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*
        )
        \.[a-zA-Z]{2,63} # TLD
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

# ===== Preprocessing Functions =====

def separate_punctuations(text):
    text_list = []
    for character in text:
        if character in punctuation_set:
            text_list.append(f" {character} ")
        else:
            text_list.append(character)
    return ''.join(text_list)

def normalize_repeated_chars(text: str) -> str:
    """حروف تکراری پشت سر هم را به یک تا کاهش می‌دهد"""
    return re.sub(r"(.)\1{2,}", r"\1", text)

def normalize_persian_unicode(text: str) -> str:
    text = text.replace("\u0640", "")
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

# ===== BPE Tokenizer =====

class BPETokenizer:
    """توکنایزر BPE با قابلیت word-level و subword-level برای فارسی"""

    def __init__(self, vocab: Dict[str, int] = None):
        self.vocab = vocab if vocab else {}
        self.bpe_merges = {}

    def get_vocab(self):
        return self.vocab

    def train_bpe(self, texts: List[str], num_merges: int = 1000):
        """آموزش BPE از متن"""
        from collections import Counter
        tokens = []
        for text in texts:
            text_clean = clean_text(text)
            text_clean = separate_punctuations(text_clean)
            tokens.extend(text_clean.split())

        vocab = Counter(tokens)
        vocab = {word + '</w>': freq for word, freq in vocab.items()}  # end-of-word symbol

        merges = {}
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            merges[best] = i
        self.vocab = vocab
        self.bpe_merges = merges

    def get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """شمارش جفت‌های متوالی"""
        stats = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                stats[pair] = stats.get(pair, 0) + freq
        return stats

    def merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """ادغام جفت‌ها در واژه‌نامه"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab

    def encode(self, text: str) -> List[str]:
        text_clean = clean_text(text)
        text_clean = separate_punctuations(text_clean)
        tokens = text_clean.split()
        subwords = []
        for token in tokens:
            token += '</w>'
            i = 0
            while i < len(token):
                matched = False
                for merge in sorted(self.bpe_merges.keys(), key=lambda x: -self.bpe_merges[x]):
                    merged = ''.join(merge)
                    if token[i:].startswith(merged):
                        subwords.append(merged)
                        i += len(merged)
                        matched = True
                        break
                if not matched:
                    subwords.append(token[i])
                    i += 1
        return subwords

    def decode(self, subwords: List[str]) -> str:
        """بازگرداندن متن اصلی"""
        text = ''.join(subwords).replace('</w>', ' ')
        return text.strip()

def tokenize(text: str) -> List[str]:
    """توکنایزر word-level برای فارسی مبتنی بر فاصله و علائم نگارشی"""
    text = clean_text(text)
    text = separate_punctuations(text)
    tokens = text.split()
    return tokens
