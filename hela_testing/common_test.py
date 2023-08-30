import unittest
import math
from Hela.common.common import Sigmoid, Gaussian, ReLU


class SigmoidTest(unittest.TestCase):
    def test_sigmoid_function(self):
        value = [-1.0, 1.0, 2.0]
        expected_result = list(1 / (1 + math.exp(-vector)) for vector in value)
        actual_result = Sigmoid(value).calculate_sigmoid()
        self.assertEqual(actual_result, expected_result)


class GaussianTest(unittest.TestCase):
    def test_calculate_gaussian(self):
        gaussian_calc = Gaussian(x=0.0)
        result = gaussian_calc.calculate_gaussian()
        self.assertAlmostEqual(result, 0.3989422804014337, places=6)

    def test_custom_parameter(self):
        gaussian_calc = Gaussian(x=2.0, mu=1.0, sigma=0.5)
        result = gaussian_calc.calculate_gaussian()
        self.assertAlmostEqual(result, 0.3441465248732082, places=6)

    def test_repr(self):
        gaussian_calc = Gaussian(x=2.0, mu=1.0, sigma=0.5)
        representation = repr(gaussian_calc)
        self.assertEqual(representation, "Gaussian(x=2.0, mu=1.0, sigma=0.5)")


class ReLU(unittest.TestCase):
    def Test_ReLU(self):
        vector = [1, 2, 3, 4]
        result = ReLU(vector).calculate_ReLu(vector)
        expeted_value = [1, 2, 3, 4]
        self.assertEqual(result.any(), expeted_value.any())
