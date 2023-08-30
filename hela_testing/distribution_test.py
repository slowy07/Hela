import unittest
import numpy as np
from Hela.common.distribution import (
    NormalDistribution,
    ContinousDistribution,
    BetaDistribution,
    ExponetialDistribution,
    DirichletDistribution,
    ChiSquareDistribution,
    Poisson,
    student_distribution,
)


class TestNormalDistribution(unittest.TestCase):
    def test_normal_pdf(self):
        pdf = NormalDistribution.normal_pdf(2.5, mean=3.0, sigma=0.8)
        self.assertAlmostEqual(pdf, 0.41020121068796883)

    def test_standard_normal_pdf(self):
        pdf = NormalDistribution.standard_normal_pdf(2.5)
        self.assertAlmostEqual(pdf, 0.01752830049356854)


class TestContinuousDistribution(unittest.TestCase):
    def test_continous_uniform_pdf(self):
        pdf = ContinousDistribution.continous_uniform_pdf(3.0, 2.0, 5.0)
        self.assertAlmostEqual(pdf, 0.333333, places=6)

    def test_continous_uniform_cdf(self):
        cdf = ContinousDistribution.continous_uniform_pdf(3.0, 2.0, 5.0)
        self.assertAlmostEqual(cdf, 0.333333, places=6)

    def test_generate_random(self):
        sample = ContinousDistribution.generate_random_sample(2.0, 5.0)
        self.assertTrue(2.0 <= sample <= 5.0)


class TestBetaDistribution(unittest.TestCase):
    def test_beta_pdf(self):
        pdf = BetaDistribution.beta_pdf(0.3, alpha=2, beta=5)
        self.assertAlmostEqual(pdf, 2.1608999999999994, places=6)

    def test_beta_cdf(self):
        cdf = BetaDistribution.beta_cdf(0.3, alpha=2, beta=5)
        self.assertAlmostEqual(cdf, 0.5787416212509751, places=6)


class TestExponentialDistribution(unittest.TestCase):
    def test_exponential_pdf(self):
        pdf = ExponetialDistribution.exponential_pdf(2.5, lambd=0.8)
        self.assertAlmostEqual(pdf, 0.10826822658929017)

    def test_exponential_cdf(self):
        cdf = ExponetialDistribution.exponential_cdf(2.5, lambd=0.8)
        self.assertAlmostEqual(cdf, 0.8646647167633873)


class TestDirichletDistribution(unittest.TestCase):
    def test_dirichlet_pdf(self):
        pdf = DirichletDistribution.dirichlet_pdf([0.3, 0.4, 0.3], alpha=[2, 3, 2])
        self.assertAlmostEqual(pdf, 5.184)

    def test_dirichlet_pdf_different_length(self):
        with self.assertRaises(ValueError):
            DirichletDistribution.dirichlet_pdf([0.3, 0.3], alpha=[2])


class TestChiSquaredDistribution(unittest.TestCase):
    def test_chi_squared_pdf(self):
        pdf = ChiSquareDistribution.chi_squared_pdf(2.5, k=3)
        self.assertAlmostEqual(pdf, 0.18072239266818124)

    def test_chi_squared_cdf(self):
        cdf = ChiSquareDistribution.chi_squared_cdf(2.5, df=2)
        self.assertAlmostEqual(cdf, 0.48982412914811513)


class TestingPoisson(unittest.TestCase):
    def test_poisson_pdf(self):
        pmf = Poisson.poisson_pmf(x=5, alpha=2)
        self.assertAlmostEqual(pmf, 0.03608940886309672)

    def test_poisson_cdf(self):
        cdf = Poisson.poisson_cdf(5, alpha=2)
        expected_result = np.array(
            [0.13533528, 0.27067057, 0.27067057, 0.18044704, 0.09022352]
        )
        self.assertAlmostEqual(cdf.any(), expected_result.any())


class Test_Student_distribution(unittest.TestCase):
    def test_T_Distribution_pdf(self):
        vector = np.array([1, 2, 3, 4])
        pdf = student_distribution.t_distribution_pdf(vector)
        expected_result = np.array([0.61917584, 8.00552478, 8.00552478, 0.61917584])
        self.assertAlmostEqual(pdf.any(), expected_result.any())
