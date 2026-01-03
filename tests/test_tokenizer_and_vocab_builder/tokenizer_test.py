import unittest
from ...tokenizer.tokenizer import tokenize


class Tests(unittest.TestCase):
    
    def test_tokenizer(self):
        """Check tokenize function"""
        self.assertEqual(tokenize("سلام, حالت چطوره؟ خوب هستی انشاالله؟ ممنون. منم خوبم! اسمت چیه؟ آهااااان. اسم منم مانی کهزادی هست."), ['سلام', 'حالت', 'چطوره', 'خوب', 'هستی', 'انشاالله', 'ممنون', 'منم', 'خوبم', 'اسمت', 'چیـه', 'آهان', 'اسم', 'منم', 'مانی', 'کهزادی', 'هست'])


if __name__ == "__main__":
    unittest.main()