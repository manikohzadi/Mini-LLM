"""
Word2Vec Skip-gram + Negative Sampling — Persian
پیاده‌سازی کامل با پشتیبانی از زبان فارسی
"""

import math # برای تابع نمایی, جذر و لگاریتم
import random # برای تولید اعداد تصادفی
import re # برای تولید الگوی توکن سازی
import json # برای ذخیره embedding
import unicodedata # برای دسترسی به کاراکتر های یونیکد
from collections import Counter # برای شمارش فرکانس یا تعداد کلمات
from typing import Dict, List, Tuple, Iterator, Optional, Set # برای type hinting که کیفیت کد رو میبره بالا


# 1) نرمال‌سازی متون فارسی
class PersianNormalizer:
    """
    حل مشکلات یونیکدی زبان فارسی.
    تمام متون را به شکل استاندارد و یکدست تبدیل می‌کند.
    """

    # تبدیل کاراکتر های عربی یا غیر استاندارد به کاراکتر های استاندارد
    CHAR_MAP = {
        'ي': 'ی',  # ی عربی
        'ى': 'ی',  # الف مقصوره
        'ئ': 'ی',  # همزه روی ی
        'ك': 'ک',  # ک عربی
        'ة': 'ه',  # ت مربوطه
        'ؤ': 'و',  # همزه روی و
        'إ': 'ا',  # همزه زیر ا
        'أ': 'ا',  # همزه روی ا
        'ئ': 'ی',  # همزه روی ی
        'ۂ': 'ه',  # همزه روی ه
        'ء': '',   # همزه تنها (معمولا حذف می‌ شود)
    }

    # اعراب عربی
    DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]') # تشکیل، تنوین‌ها، سکون، تشدید

    # الگوی فاصله‌ های غیر استاندارد
    WEIRD_SPACES = re.compile(r'[\u2000-\u200B\u2028\u2029\u202F\u205F\u3000]')

    # نیم‌ فاصله (Zero-Width Non-Joiner)
    ZWNJ = '\u200c'

    @classmethod # نیازی به instance نداریم و می توانیم مستقیم صدا بزنیمش.
    def normalize(cls, text: str) -> str:
        """اجرای تمام مراحل نرمال‌ سازی به ترتیب."""

        # مرحله 1 : نرمال‌ سازی یونیکد NFC
        text = unicodedata.normalize('NFC', text) # تمام کاراکتر ها را به حالت ترکیب شده (Normalization Form Composed) تبدیل می کنیم و اگر چنین کاری نکنیم بعدا نگاشت (CHAR_MAP) ما به درستی کار نمی کند

        # مرحله 2 : تبدیل کاراکتر های عربی
        for old, new in cls.CHAR_MAP.items():
            text = text.replace(old, new) # نگاشت کاراکتر ها

        # مرحله 3 : حذف اعراب
        # اعراب در متن فارسی مدرن کاربردی ندارد
        text = cls.DIACRITICS.sub('', text)

        # مرحله 4 : حذف کشیده (تطویل)
        text = text.replace('ـ', '')

        # مرحله 5 : یکدست‌ سازی فاصله‌ ها
        text = cls.WEIRD_SPACES.sub(' ', text) # تبدیل فاصله های عجیب به عادی
        text = re.sub(r'\s+', ' ', text).strip() # یک یا چند فاصله از هر نوع را با یک فاصله جایگزین می کنیم. بعد فاصله های ابتدایی و انتهایی را قیچی می کنیم

        # مرحله 6 : پاک‌ سازی نیم‌ فاصله
        text = re.sub(rf'{cls.ZWNJ}+', cls.ZWNJ, text) # چند نیم‌ فاصله پشت سر هم تبدیل می شود به یکی

        text = re.sub(rf' {cls.ZWNJ}', cls.ZWNJ, text) # نیم‌ فاصله نباید بعد از فاصله یا قبل از فاصله بیاید
        text = re.sub(rf'{cls.ZWNJ} ', ' ', text)
        
        text = text.strip(cls.ZWNJ) # نباید در ابتدا یا انتهای متن باشد

        return text


# 2) توکن‌ سازی فارسی
class PersianTokenizer:
    """
    توکن‌ سازی که نیم‌ فاصله را داخل کلمه حفظ می‌کند.
    مثلا "می‌رود" یک توکن است, نه دو توکن.
    """

    PERSIAN_LETTERS = r'\u0600-\u06FF' # بلوک اصلی فارسی/عربی
    ARABIC_FORMS = r'\uFB50-\uFDFF\uFE70-\uFEFF' # اشکال نمایشی
    ZWNJ = r'\u200c' # نیم فاصله

    # الگوی توکن‌ سازی:
    # 1) ;gli
    # 2) یا کلمه انگلیسی
    # 3) یا عدد
    # 4) یا نویسه تکی (نقطه‌ گذاری)
    TOKEN_PATTERN = re.compile(
        '[' + PERSIAN_LETTERS + ARABIC_FORMS + ZWNJ + ']+' # کلمه فارسی
        '|[A-Za-z]+' # کلمه انگلیسی
        '|\d+' # عدد
        '|[^\s\w]' # نویسه تکی غیر فاصله و عیر حروف/عدد (نقطه گذاری)
    )

    def __init__(self, min_length: int = 2):
        self.min_length = min_length

    def tokenize(self, text: str) -> List[str]:
        tokens = self.TOKEN_PATTERN.findall(text)
        # حذف توکن‌های کوتاه
        return [
            t for t in tokens
            if len(t) >= self.min_length and t.strip()
        ]


# ============================================================
# 3) Persian Preprocessor — رابط اصلی برای استفاده
# ============================================================
class PersianPreprocessor:
    """
    نقطه ورود اصلی برای پیش‌پردازش متون فارسی.
    ترکیب نرمال‌ساز + توکن‌ساز.
    """

    def __init__(self, min_length: int = 2, normalize: bool = True):
        self.min_length = min_length
        self.normalize = normalize
        self.normalizer = PersianNormalizer()
        self.tokenizer = PersianTokenizer(min_length)

    def preprocess(self, text: str) -> List[str]:
        """تبدیل یک سند به لیست توکن‌ها."""
        if self.normalize:
            text = self.normalizer.normalize(text)
        return self.tokenizer.tokenize(text)

    def preprocess_corpus(self, texts: List[str]) -> List[List[str]]:
        """پیش‌پردازش چندین سند."""
        return [self.preprocess(t) for t in texts]

    def preprocess_file(self, filepath: str,
                        encoding: str = 'utf-8') -> Iterator[List[str]]:
        """پیش‌پردازش خط به خط (برای فایل‌های بزرگ)."""
        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    yield self.preprocess(line)


# ============================================================
# 4) Vocabulary — بدون تغییر نسبت به نسخه انگلیسی
# ============================================================
class Vocabulary:
    def __init__(self, min_count: int = 5):
        self.min_count = min_count
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.counts: List[int] = []
        self.total_words: int = 0

    def build(self, corpus: List[List[str]]) -> None:
        counter = Counter()
        for doc in corpus:
            counter.update(doc)

        sorted_words = sorted(counter.items(), key=lambda x: (-x, x))

        self.word2idx, self.idx2word, self.counts = {}, [], []
        for word, count in sorted_words:
            if count >= self.min_count:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
                self.counts.append(count)

        self.total_words = sum(self.counts)
        print(f"  ✓ vocab size: {len(self.idx2word):,} | "
              f"total tokens: {self.total_words:,}")

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.word2idx[t] for t in tokens if t in self.word2idx]

    @property
    def size(self) -> int:
        return len(self.idx2word)


# ============================================================
# 5) Subsampler — بدون تغییر
# ============================================================
class Subsampler:
    def __init__(self, vocab: Vocabulary, threshold: float = 1e-5):
        self.threshold = threshold
        self.keep_prob: Dict[int, float] = {}
        total = vocab.total_words
        for idx, count in enumerate(vocab.counts):
            f = count / total
            self.keep_prob[idx] = (math.sqrt(f / threshold) + 1) * (threshold / f)

    def subsample(self, encoded: List[int], rng: random.Random) -> List[int]:
        return [idx for idx in encoded if rng.random() < self.keep_prob[idx]]


# ============================================================
# 6) NegativeSampler — بدون تغییر
# ============================================================
class NegativeSampler:
    def __init__(self, vocab: Vocabulary, power: float = 0.75):
        counts_pow = [c ** power for c in vocab.counts]
        total = sum(counts_pow)
        self.cumsum: List[float] = []
        acc = 0.0
        for c in counts_pow:
            acc += c / total
            self.cumsum.append(acc)
        self.vocab_size = vocab.size

    def sample(self, k: int, exclude: Set[int], rng: random.Random) -> List[int]:
        samples, attempts = [], 0
        while len(samples) < k and attempts < k * 20:
            r = rng.random()
            lo, hi = 0, self.vocab_size - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self.cumsum[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            if lo not in exclude:
                samples.append(lo)
            attempts += 1
        return samples


# ============================================================
# 7) PairGenerator — بدون تغییر
# ============================================================
class PairGenerator:
    def __init__(self, max_window: int = 5, dynamic: bool = True):
        self.max_window = max_window
        self.dynamic = dynamic

    def generate(self, seq: List[int], rng: random.Random) -> Iterator[Tuple[int, int]]:
        n = len(seq)
        for i, center in enumerate(seq):
            w = rng.randint(1, self.max_window) if self.dynamic else self.max_window
            for j in range(max(0, i - w), min(n, i + w + 1)):
                if i != j:
                    yield center, seq[j]


# ============================================================
# 8) Word2Vec Model — بدون تغییر (زبان-مستقل است!)
# ============================================================
class Word2Vec:
    def __init__(self,
                 vector_size: int = 100,
                 max_window: int = 5,
                 negative_samples: int = 5,
                 min_count: int = 5,
                 subsample_threshold: float = 1e-5,
                 learning_rate: float = 0.05,
                 min_learning_rate: float = 0.0001,
                 epochs: int = 5,
                 seed: int = 42):
        self.vector_size = vector_size
        self.max_window = max_window
        self.negative_samples = negative_samples
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.epochs = epochs
        self.seed = seed

        self.vocab: Optional[Vocabulary] = None
        self.input_vectors: Dict[int, List[float]] = {}
        self.output_vectors: Dict[int, List[float]] = {}

    def _init_vectors(self, vocab_size: int, rng: random.Random) -> None:
        scale = 0.5 / self.vector_size
        self.input_vectors = {
            i: [(rng.random() - 0.5) * 2 * scale for _ in range(self.vector_size)]
            for i in range(vocab_size)
        }
        self.output_vectors = {
            i: [0.0] * self.vector_size
            for i in range(vocab_size)
        }

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _train_pair(self, center: int, context: int, lr: float,
                    neg_sampler: NegativeSampler, rng: random.Random) -> float:
        v_c = self.input_vectors[center]
        u_o = self.output_vectors[context]

        score = self._dot(u_o, v_c)
        score = max(-10.0, min(10.0, score))
        sigmoid_pos = 1.0 / (1.0 + math.exp(-score))
        grad_pos = sigmoid_pos - 1.0
        loss = -math.log(max(sigmoid_pos, 1e-10))

        for i in range(self.vector_size):
            u_o[i] -= lr * grad_pos * v_c[i]
            v_c[i] -= lr * grad_pos * u_o[i]

        for neg in neg_sampler.sample(self.negative_samples,
                                      exclude={center, context}, rng=rng):
            u_n = self.output_vectors[neg]
            score_n = self._dot(u_n, v_c)
            score_n = max(-10.0, min(10.0, score_n))
            sigmoid_neg = 1.0 / (1.0 + math.exp(-score_n))
            grad_neg = sigmoid_neg
            loss += -math.log(max(1 - sigmoid_neg, 1e-10))

            for i in range(self.vector_size):
                u_n[i] -= lr * grad_neg * v_c[i]
                v_c[i] -= lr * grad_neg * u_n[i]

        return loss

    def fit(self, corpus: List[List[str]], verbose: bool = True) -> None:
        rng = random.Random(self.seed)

        if verbose: print("\n[1/4] ساخت واژگان...")
        self.vocab = Vocabulary(self.min_count)
        self.vocab.build(corpus)
        if self.vocab.size == 0:
            raise ValueError("واژگان خالی — min_count را کم کنید")

        if verbose: print(f"[2/4] مقداردهی اولیه {self.vocab.size:,} بردار...")
        self._init_vectors(self.vocab.size, rng)

        subsampler = Subsampler(self.vocab, self.subsample_threshold)
        neg_sampler = NegativeSampler(self.vocab)
        pair_gen = PairGenerator(self.max_window, dynamic=True)

        if verbose: print("[3/4] تبدیل متن به اعداد...")
        encoded_corpus = [self.vocab.encode(doc) for doc in corpus]

        if verbose: print(f"[4/4] آموزش برای {self.epochs} دوره...\n")
        for epoch in range(self.epochs):
            lr = max(self.min_learning_rate,
                     self.learning_rate * (1 - epoch / self.epochs))
            rng.shuffle(encoded_corpus)

            pair_count, loss_sum = 0, 0.0
            report_every = 50_000

            for doc in encoded_corpus:
                doc = subsampler.subsample(doc, rng)
                if len(doc) < 2:
                    continue
                for c, ctx in pair_gen.generate(doc, rng):
                    loss_sum += self._train_pair(c, ctx, lr, neg_sampler, rng)
                    pair_count += 1
                    if verbose and pair_count % report_every == 0:
                        print(f"  دوره {epoch+1} | جفت‌ها: {pair_count:>10,} | "
                              f"loss: {loss_sum/report_every:.4f}")
                        loss_sum = 0.0
            if verbose:
                print(f"  ✓ دوره {epoch+1}/{self.epochs} تمام | "
                      f"جمع جفت‌ها: {pair_count:,}\n")

    def get_vector(self, word: str) -> Optional[List[float]]:
        if self.vocab is None or word not in self.vocab.word2idx:
            return None
        return self.input_vectors[self.vocab.word2idx[word]][:]

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        v = self.get_vector(word)
        if v is None: return []
        norm = math.sqrt(self._dot(v, v))
        if norm == 0: return []
        v = [x / norm for x in v]

        scores = []
        for idx in range(self.vocab.size):
            w = self.vocab.idx2word[idx]
            if w == word: continue
            u = self.input_vectors[idx]
            u_norm = math.sqrt(self._dot(u, u))
            if u_norm == 0: continue
            scores.append((w, self._dot(v, u) / u_norm))
        scores.sort(key=lambda x: -x)
        return scores[:topn]

    def analogy(self, a: str, b: str, c: str, topn: int = 5):
        va, vb, vc = self.get_vector(a), self.get_vector(b), self.get_vector(c)
        if None in (va, vb, vc): return []
        target = [b_ - a_ + c_ for a_, b_, c_ in zip(va, vb, vc)]
        norm = math.sqrt(self._dot(target, target))
        target = [x / norm for x in target]

        exclude = {a, b, c}
        scores = []
        for idx in range(self.vocab.size):
            w = self.vocab.idx2word[idx]
            if w in exclude: continue
            u = self.input_vectors[idx]
            u_norm = math.sqrt(self._dot(u, u))
            if u_norm == 0: continue
            scores.append((w, self._dot(target, u) / u_norm))
        scores.sort(key=lambda x: -x)
        return scores[:topn]

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"{self.vocab.size} {self.vector_size}\n")
            for idx, word in enumerate(self.vocab.idx2word):
                combined = [(a + b) / 2
                            for a, b in zip(self.input_vectors[idx],
                                            self.output_vectors[idx])]
                f.write(word + ' ' + ' '.join(f"{v:.6f}" for v in combined) + '\n')