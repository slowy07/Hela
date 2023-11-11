import unittest
from Hela.common.antidifferential import AntiDifferential
from math import sqrt


class Testing_AntiDifferential(unittest.TestCase):
    def setUp(self):
        self.intergral = AntiDifferential()

    def test_antidifferential_RiemannSum(self):
        formula = lambda x: x**2
        result = self.intergral.general_antiderivative(formula, 1, 4, 100)
        self.assertEqual(result, 21.0)

    def test_antidifferential_trapezoidal(self):
        formula = lambda x: x**2
        result = self.intergral.general_antiderivative(
            formula, 0, 3, method="trapezoidal"
        )
        self.assertEqual(result, 9.0)

    def testing_powerRule_antidifferential(self):
        result = self.intergral.PowerRule_antiderivative(n=3, x=2)
        self.assertEqual(result, 4.0)

    def testing_Usubtitution(self):
        g_formula = lambda x: sqrt(x**4 + 11)
        f_formula = lambda x: x**3 * g_formula(x)
        result = AntiDifferential.Usubstitution(f_formula, g_formula, 1, 2)
        self.assertEqual(result, -30.31)
