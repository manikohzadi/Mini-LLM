"""
HTML/XML tag related patterns.

These patterns are designed for text sanitization and tag removal,
NOT for parsing HTML or XML documents.
"""


import re

# HTML / XML Comments

HTML_COMMENT_PATTERN = re.compile(
    r"""
    <!-- # 
    .*?
    -->
    """,
    re.VERBOSE | re.DOTALL,
)

# DOCTYPE

DOCTYPE_PATTERN = re.compile(
    r"""
    <!DOCTYPE
    \s+
    [^>]+
    >
    """,
    re.VERBOSE | re.IGNORECASE,
)

# XML Declaration

XML_DECLARATION_PATTERN = re.compile(
    r"""
    <\?xml
    \s+
    .*?
    \?>
    """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)

# CDATA

CDATA_PATTERN = re.compile(
    r"""
    <!\[CDATA\[
    .*?
    \]\]>
    """,
    re.VERBOSE | re.DOTALL,
)

# Processing Instructions
# (except XML declaration)

PROCESSING_INSTRUCTION_PATTERN = re.compile(
    r"""
    <\?
    (?!xml\b)
    .*?
    \?>
    """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)

# HTML / XML Tag

HTML_TAG_PATTERN = re.compile(
    r"""
    <
        /?
        [A-Za-z][A-Za-z0-9:-]*

        (?:
            \s+

            [^\s<>=/"']+

            (?:
                \s*=\s*
                (?:
                    "[^"]*"
                    |
                    '[^']*'
                    |
                    [^\s"'=<>`]+
                )
            )?
        )*

        \s*
        /?
    >
    """,
    re.VERBOSE,
)

# HTML Entity

HTML_ENTITY_PATTERN = re.compile(
    r"""
    &
    (?:
        [A-Za-z][A-Za-z0-9]+
        |
        \#[0-9]+
        |
        \#x[0-9A-Fa-f]+
    )
    ;
    """,
    re.VERBOSE,
)