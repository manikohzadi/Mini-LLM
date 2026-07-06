"""
پیاده سازی الگو هایی برای شناسایی انواع تگ های HTML/XML.

این الگو ها در اعتبارسنجی تگ ها نمی کوشند بلکه می خواهد هر چیزی که تقریبا شبیه تگ است را از متن حذف کنند.
"""


import re

# HTML / XML Comments

HTML_COMMENT_PATTERN = re.compile(
    r"""
    <!-- # دقیقا همین متن
    .*?
    # . : یعنی هر کاراکتری و چون فلگ DOTALL وجود دارد حتی \n هم حساب می شود
    # .* : هر نعداد کاراکتر(صفر یا بیشتر)
    # .*? : اما هنوز یک مشکل داریم و اینکه اگر ? نذاریم رجکس Greedy میشه و مثلا در متن پایین:
    #   <!-- first -->
    #   text
    #   <!-- second -->
    # کل متن match می شود چون تا آخرین --> می رود که درست نیست
    # حالا وجود ? بعد از * باعث می‌شود تکرار Lazy شود.
    # یعنی : تا اولین جایی که ادامه Regex قابل تطبیق باشد, پیش برو.
    --> # پایان کامنت

    # کل الگو میشه : از اولین <!-- تا اولین -->.
    """,
    re.VERBOSE | re.DOTALL,
)

# DOCTYPE

DOCTYPE_PATTERN = re.compile(
    r"""
    <!DOCTYPE # دقیقا همین متن
    \s+ # حداقل یک فاصله
    [A-Za-z][^\s>]* # اولین کاراکتر Name باید یک حرف انگلیسی باشد و ادامه Name باید هر کاراکتری به جز فاصله و > باشد. و ممکن هم هست Name تک حرفی باشه
    (?:\s+[^>]*)? # این بخش اختیاری هست : حداقل یک فاصله و هر چیزی غیر از > که می تونه اصلا نباشه و فقط حداقل یک فاصله باشه
    > # پایان DOCTYPE
    """,
    re.VERBOSE | re.IGNORECASE,
)

# XML Declaration

XML_DECLARATION_PATTERN = re.compile(
    r"""
    <\?xml # خود <?xml که ? را با \ escape کردیم
    \s+ # حداقل یک فاصله
    .*? # هر چیزی(به صورت Lazy)
    \?> # یعنی ?> که پایان Declaration است
    """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)

# CDATA

CDATA_PATTERN = re.compile(
    r"""
    <!\[CDATA\[ # <![CDATA[ با escape کردن
    .*? # هر چیزی(به صورت Lazy)
    \]\]> # ]]>
    """,
    re.VERBOSE | re.DOTALL,
)

# Processing Instructions
# (except XML declaration)

PROCESSING_INSTRUCTION_PATTERN = re.compile(
    r"""
    <\? # <?
    (?!xml\b) # اینجا وارد lookahead می شویم یعنی بعد از <? نباید xml باشد چون چون XML Declaration را قبلا جدا گرفته‌ ایم
    # \b مرز کلمه(word boundary) است که یعنی xml دقیقا کلمه باشد نه مثلا xmlmani
    .*? # هر چیزی(به صورت Lazy)
    \?> # ?>
    """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)

# HTML / XML Tag

HTML_TAG_PATTERN = re.compile(
    r"""
    < # علامت شروع تگ
        /? # اسلش می تونه باشه یا نباشه
        [A-Za-z][A-Za-z0-9:-]* # اولین کاراکتر نام تگ حتما باید حرف انگلیشسی باشد بعد هم ادامه نام تگ که شامل حروف و اعداد و - و : می تواند باشد
        # یا می تواند تک حرفی باشد نام تگ

        (?: # شروع گروه Non-Capturing برای شناسایی attribute
            \s+ # حداقل یک فاصله چون attribute بدون فاصله ممکن نیست

            [^\s<>=/"']+ # نام attribute که کاراکتر های مقابل برای نوشتن آن ممنوع است

            (?: # ممکن است attribute مقدار داشته باشد یا نداشته باشد
                \s*=\s* # علامت مساوی که می تونه قبل و بعدش به هر تعداد فاصله باشه
                (?: # برای مقدار سه حالت داریم
                    "[^"]*" # رشته داخل دابل کوتیشن
                    |
                    '[^']*' # رشته داخل سینگل کوتیشن
                    |
                    [^\s"'=<>`]+ # attribute بدون quote
                )
            )?
        )* # هر تعداد attribute

        \s* #فاصله های انتهایی که میتونه نباشه
        /? # اسلش انتهایی مثل <img /> که اختیاری هست
    > # پایان تگ
    """,
    re.VERBOSE,
)

# HTML Entity

HTML_ENTITY_PATTERN = re.compile(
    r"""
    & # شروع entity
    (?: # سه حالت
        [A-Za-z][A-Za-z0-9]+ # برای مثلا &amp;
        |
        \#[0-9]+ # &#160;
        |
        \#x[0-9A-Fa-f]+ # &#xA0;
    )
    ; # پایان entity
    """,
    re.VERBOSE,
)

# combining the patterns

TAG_PATTERNS = (
    HTML_COMMENT_PATTERN,
    DOCTYPE_PATTERN,
    XML_DECLARATION_PATTERN,
    CDATA_PATTERN,
    PROCESSING_INSTRUCTION_PATTERN,
    HTML_TAG_PATTERN,
    HTML_ENTITY_PATTERN,
)

TAG_PATTERN = re.compile(
    "|".join(
        f"(?:{pattern.pattern})"
        for pattern in TAG_PATTERNS
    ),
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)