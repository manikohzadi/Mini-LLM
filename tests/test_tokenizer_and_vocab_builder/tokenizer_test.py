import unittest
from tokenizer.tokenizer import tokenize


class TokenizerIndustrialTests(unittest.TestCase):

    def test_basic_persian_sentence(self):
        tokens = tokenize("سلام، حالت چطوره؟")
        self.assertEqual(tokens, ["سلام", "،", "حالت", "چطوره", "؟"])

    def test_no_empty_tokens(self):
        tokens = tokenize("   سلام    دنیا   ")
        self.assertTrue(all(token.strip() for token in tokens))

    def test_repeated_char_normalization(self):
        tokens = tokenize("سلااااااااام")
        self.assertIn("سلام", tokens)

    def test_punctuation_isolated(self):
        tokens = tokenize("سلام!!!خوبی؟؟")
        self.assertIn("!", tokens)
        self.assertIn("؟", tokens)
        self.assertNotIn("!!!", tokens)

    def test_unicode_half_space(self):
        tokens = tokenize("می\u200cروم")
        self.assertIn("میروم", tokens)

    def test_tatweel_removed(self):
        tokens = tokenize("چیـه")
        self.assertIn("چیه", tokens)

    def test_html_removed(self):
        tokens = tokenize("<b>سلام</b>")
        self.assertEqual(tokens, ["سلام"])

    def test_url_removed(self):
        tokens = tokenize("سلام https://test.com")
        self.assertEqual(tokens, ["سلام"])

    def test_emoji_removed(self):
        tokens = tokenize("سلام 😂😂")
        self.assertEqual(tokens, ["سلام"])

    def test_deterministic(self):
        text = "سلام دنیا"
        self.assertEqual(tokenize(text), tokenize(text))

    def test_mixed_language(self):
        tokens = tokenize("سلام hello دنیا 123")
        self.assertIn("سلام", tokens)
        self.assertIn("hello", tokens)

    def test_no_whitespace_inside_token(self):
        tokens = tokenize("سلام دنیا")
        for t in tokens:
            self.assertNotRegex(t, r"\s")


if __name__ == "__main__":
    unittest.main()
