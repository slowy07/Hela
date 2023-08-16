import unittest
from Hela.math import Math


class MathTest(unittest.TestCase):
    math_obj = Math()

    # sin test
    def test_factorial_positive(self):
        result = self.math_obj.factorial(5)
        expected = 120
        self.assertEqual(result, expected)

    def test_factorial_zero(self):
        result = self.math_obj.factorial(0)
        expected = 1
        self.assertEqual(result, expected)

    def test_sine_positive(self):
        result = self.math_obj.sine(180.0)
        expected = 0
        self.assertEqual(expected, result)

    def test_sine_negative(self):
        result = self.math_obj.sine(-689.0)
        expected = 0.5150380749
        self.assertEqual(expected, result)

    # testing cosine
    def test_cosine_positive(self):
        result = self.math_obj.cosine(0)
        expected = 1
        self.assertEqual(expected, result)

    # average mean test
    def test_mean_with_positive_number(self):
        result = self.math_obj.avg_mean([3, 6, 9, 12, 15, 18, 21])
        expected = 12.0
        self.assertEqual(result, expected)

    def test_exponential(self):
        result = self.math_obj.exponential(2)
        expected = 7.3890560989
        self.assertEqual(result, expected)
