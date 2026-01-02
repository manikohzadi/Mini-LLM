import unittest


class Tests(unittest.TestCase):
    
    def test_1(self):
        """Check GitHub Actions"""
        self.assertTrue(2 + 2 == 4)


if __name__ == "__main__":
    unittest.main()